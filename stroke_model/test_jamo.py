"""Fresh-deformation and truly unseen-font tests for a frozen Jamo checkpoint."""
import hashlib
import io
import itertools
import json
import random
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .convert_reference_csv import existing_path, normalized
from .prepare_jamo import scan_jamo, read_components, shared_options, transform_mask, make_labels, compose
from .preview_reference import preview_reference
from .reference_model import ReferenceStrokeModel
from .train_reference import read_manifest, StrokeDataset, collate

PROFILES = {
    'standard': dict(scale_range=(.8, 1.), rotation=5., shift_fraction=20/256),
    'strong': dict(scale_range=(.65, 1.15), rotation=15., shift_fraction=40/256),
}


def image_digest(image):
    return hashlib.sha256(np.ascontiguousarray(image, dtype=np.float32).tobytes()).hexdigest()


def select_test_fonts(fonts, provenance):
    used = set(provenance['train_fonts']) | set(provenance['validation_fonts'])
    return sorted(set(fonts) & set(provenance['train_fonts'])), sorted(set(fonts) - used)


def save_training_inventory(manifest, output, training_archive, settings):
    """Persist the actual train/val image hashes, including optional augmentations."""
    rows = read_manifest(manifest)
    fingerprints = set()
    for row in tqdm(rows, desc='Save training provenance', unit='sample'):
        with np.load(row['path'], allow_pickle=False) as sample:
            fingerprints.add(image_digest(sample['image']))
    payload = dict(train_fonts=sorted({r['font'] for r in rows if r['split'] == 'train'}),
                   validation_fonts=sorted({r['font'] for r in rows if r['split'] == 'val'}),
                   fingerprints=sorted(fingerprints), training_seed=settings['seed'],
                   training_archive=str(training_archive), source=settings['source'], version=1)
    (Path(output) / 'training_inventory.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def training_inventory(checkpoint, archive=None, local_root=None):
    """Recover train/validation provenance from the exact saved training ZIP."""
    checkpoint = existing_path(checkpoint)
    saved_inventory = checkpoint.parent / 'training_inventory.json'
    if not archive and saved_inventory.is_file():
        payload = json.loads(saved_inventory.read_text(encoding='utf-8'))
        if payload.get('version') != 1 or not payload.get('fingerprints'):
            raise ValueError('Invalid saved training inventory')
        payload['fingerprints'] = set(payload['fingerprints'])
        print('Read saved training image/font inventory:', saved_inventory, flush=True)
        return payload
    config_path = checkpoint.parent / 'run_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    if not archive:
        archive = config.get('training_archive')
    if not archive:
        if not config:
            raise ValueError('No run_config.json; specify TRAINING_ARCHIVE explicitly to identify train/val fonts.')
        matches = []
        for candidate in sorted((checkpoint.parent.parent / 'data').glob('jamo_*.zip')):
            with zipfile.ZipFile(candidate) as bundle:
                report = json.loads(bundle.read('conversion_report.json'))
            settings = report.get('settings', {})
            if settings.get('version') != 4 or report.get('reference_mode') != 'same_font':
                continue
            required = ('seed', 'samples_per_class', 'max_overlap')
            if any(k not in config or config[k] != settings.get(k) for k in required):
                continue
            if normalized(config.get('source_root', '')) != normalized(settings.get('source', '')):
                continue
            matches.append(candidate)
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely identify training ZIP ({len(matches)} matches). Set TRAINING_ARCHIVE to the ZIP used for this checkpoint.')
        archive = matches[0]
    archive = existing_path(archive)
    local_root = Path(local_root) if local_root else Path(tempfile.gettempdir()) / 'jamo_test'
    local_root.mkdir(parents=True, exist_ok=True)
    local_archive = local_root / ('training_' + hashlib.sha256(str(archive).encode()).hexdigest()[:12] + '.zip')
    print('Copy training ZIP once to identify fonts and reject identical images...', flush=True)
    shutil.copyfile(archive, local_archive)
    fingerprints = set()
    with zipfile.ZipFile(local_archive) as bundle:
        rows = json.loads(bundle.read('manifest.json'))
        report = json.loads(bundle.read('conversion_report.json'))
        if report.get('reference_mode') != 'same_font' or any(r.get('reference_font') != r.get('font') for r in rows):
            raise ValueError('Checkpoint data used different reference/input fonts. Select a same-font Jamo run.')
        for row in tqdm(rows, desc='Fingerprint training/validation images', unit='sample'):
            with np.load(io.BytesIO(bundle.read(row['path'])), allow_pickle=False) as sample:
                fingerprints.add(image_digest(sample['image']))
    if config.get('variants', 0) > 0:
        augmented_manifest = Path(config.get('training_manifest', ''))
        if not augmented_manifest.is_file():
            raise ValueError('This run used extra augmentations. Recover its training_manifest locally to fingerprint the actual augmented images; the base ZIP alone is insufficient.')
        for row in tqdm(read_manifest(augmented_manifest), desc='Fingerprint augmented training images'):
            with np.load(row['path'], allow_pickle=False) as sample:
                fingerprints.add(image_digest(sample['image']))
    train_fonts = sorted({r['font'] for r in rows if r['split'] == 'train'})
    val_fonts = sorted({r['font'] for r in rows if r['split'] == 'val'})
    if not train_fonts or not val_fonts or set(train_fonts) & set(val_fonts):
        raise ValueError('Invalid training/validation font provenance')
    return dict(train_fonts=train_fonts, validation_fonts=val_fonts, fingerprints=fingerprints,
                training_seed=report['settings']['seed'], training_archive=str(archive),
                source=report['settings']['source'])


def generate_test(fonts, loaded, selected_fonts, destination, seed, profile,
                  samples_per_class=30, excluded_images=frozenset(), max_overlap=.001, max_attempts=20):
    """Every reference/input pair uses the same font; no training is performed."""
    if profile not in PROFILES or samples_per_class < 1:
        raise ValueError('Invalid test generation settings')
    destination = Path(destination)
    (destination / 'samples').mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    rows, rejected = [], []
    seen_images = set(excluded_images)
    attempted = 0
    for font in tqdm(selected_fonts, desc=f'Generate {profile} test', unit='font'):
        bank = fonts[font]
        for cls in range(6):
            options = shared_options(bank, bank, cls)
            if not options:
                continue
            combinations = list(itertools.product(*options))
            rng.shuffle(combinations)
            for parts in combinations[:samples_per_class]:
                attempted += 1
                count = 2 if cls < 3 else 3
                paths = [bank[cls][slot][parts[slot]][0] for slot in range(count)]
                if len({loaded[p][1] for p in paths}) != 1:
                    raise ValueError(f'Mismatched component canvases: {font}/{cls}')
                original = np.stack([loaded[p][0] for p in paths])
                reason = 'overlap_or_empty_component'
                for attempt in range(max_attempts):
                    masks = np.stack([transform_mask(mask, rng, **PROFILES[profile]) for mask in original])
                    labels, overlap, valid = make_labels(masks, max_overlap)
                    image = masks.max(0, keepdims=True)
                    fingerprint = image_digest(image)
                    if valid and fingerprint not in seen_images:
                        break
                    if valid:
                        reason = 'identical_image'
                else:
                    rejected.append(dict(font=font, class_id=cls, char=compose(parts), reason=reason))
                    continue
                seen_images.add(fingerprint)
                path = Path('samples') / f'{len(rows):06d}.npz'
                np.savez_compressed(destination / path, image=image, reference=original.max(0, keepdims=True),
                                    strokes=original, labels=labels)
                rows.append(dict(path=path.as_posix(), split='test', group_id='font:' + font,
                                 font=font, reference_font=font, class_id=cls, char_id=compose(parts),
                                 component_slots=list(range(count)), jamo_indices=list(parts),
                                 target_components=[normalized(p.name) for p in paths],
                                 image_sha256=fingerprint, overlap_canvas_ratio=overlap, source='jamo_test'))
    if not rows:
        raise ValueError(f'No accepted {profile} test samples; inspect component banks/overlap settings.')
    manifest = destination / 'manifest.json'
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    generation = dict(seed=seed, profile=profile, transform=PROFILES[profile], max_overlap=max_overlap,
                      attempted_combinations=attempted, accepted=len(rows), rejected=rejected,
                      acceptance_rate=len(rows) / max(attempted, 1),
                      note='Repeated semantic characters are allowed; identical train/val/test input images are rejected.')
    (destination / 'generation.json').write_text(json.dumps(generation, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest, generation


def segmentation_metrics(prediction, labels, k):
    """Per-jamo macro scores on annotated pixels, plus aggregate confusion counts."""
    known = labels >= 0
    if not known.any():
        raise ValueError('No annotated pixels')
    confusion = np.bincount((labels[known] * k + prediction[known]).astype(np.int64),
                            minlength=k*k).reshape(k, k)
    tp = confusion.diagonal().astype(np.float64)
    mass = confusion.sum(1)
    fp = confusion.sum(0) - tp
    fn = mass - tp
    active = mass > 0
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
    dice = divide(2 * tp, 2 * tp + fp + fn)
    scores = dict(dice=dice, f1=dice.copy(), iou=divide(tp, tp+fp+fn),
                  precision=divide(tp, tp+fp), recall=divide(tp, tp+fn))
    return {name: values[active].tolist() for name, values in scores.items()}, dict(
        correct=int(tp.sum()), annotated=int(mass.sum()), strokes=int(active.sum()))


def reference_baseline(strokes):
    """Assign pixels to the nearest original reference-jamo foreground."""
    distances = np.stack([distance_transform_edt(mask <= .5) for mask in strokes])
    return distances.argmin(0)


@torch.no_grad()
def evaluate_test(model, manifest, device, size=128, batch_size=8):
    rows = read_manifest(manifest)
    dataset = StrokeDataset(rows, size)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    accumulated = {key: [] for key in ('dice', 'f1', 'iou', 'precision', 'recall')}
    baseline_values, details = [], []
    correct = annotated = foreground = 0
    model.eval()
    index = 0
    for batch in tqdm(loader, desc='Evaluate frozen checkpoint', unit='batch'):
        image, ref, strokes, labels = [x.to(device) for x in batch]
        predictions = model(image, ref, strokes)['probabilities'].argmax(1).cpu().numpy()
        images, originals, targets = image.cpu().numpy(), strokes.cpu().numpy(), labels.cpu().numpy()
        for i, prediction in enumerate(predictions):
            row = rows[index]
            k = len(row['component_slots'])
            scores, counts = segmentation_metrics(prediction, targets[i], k)
            for key in accumulated:
                accumulated[key].extend(scores[key])
            baseline = reference_baseline(originals[i, :k])
            baseline_scores, _ = segmentation_metrics(baseline, targets[i], k)
            baseline_values.extend(baseline_scores['dice'])
            correct += counts['correct']
            annotated += counts['annotated']
            foreground += int((images[i, 0] > .5).sum())
            details.append(dict(font=row['font'], class_id=row['class_id'], char_id=row['char_id'],
                                sample_index=index, strokes=counts['strokes'],
                                metrics={key: float(np.mean(value)) for key, value in scores.items()},
                                baseline_dice=float(np.mean(baseline_scores['dice']))))
            index += 1
    overall = {f'macro_{key}_annotated': float(np.mean(values)) for key, values in accumulated.items()}
    overall.update(foreground_accuracy_annotated=correct / annotated, annotation_coverage=annotated / max(foreground, 1),
                   reference_baseline_macro_dice=float(np.mean(baseline_values)), samples=len(rows))
    grouped = {}
    for field in ('font', 'class_id'):
        grouped[field] = {}
        for value in sorted({row[field] for row in details}, key=str):
            chosen = [row for row in details if row[field] == value]
            weights = np.array([row['strokes'] for row in chosen])
            grouped[field][str(value)] = dict(samples=len(chosen), **{
                f'macro_{key}_annotated': float(np.average([row['metrics'][key] for row in chosen], weights=weights))
                for key in accumulated})
    return dict(overall=overall, by_group=grouped, per_sample=details)


def run_jamo_test(checkpoint, source_root=None, new_font_root=None, training_archive=None,
                  seed=20261015, samples_per_class=30, preview_count=3, workers=8,
                  local_root='/content/jamo_test', batch_size=8, max_overlap=.001):
    checkpoint = existing_path(checkpoint)
    provenance = training_inventory(checkpoint, training_archive, local_root)
    if provenance['training_seed'] in range(seed, seed + 4):
        raise ValueError('Test seed must differ from the training seed.')
    source_root = source_root or provenance['source']
    _, fonts = scan_jamo(source_root, minimum_fonts=1)
    if new_font_root:
        _, additional = scan_jamo(new_font_root, minimum_fonts=1)
        if set(additional) & set(fonts):
            raise ValueError('NEW_FONT_ROOT contains font names already present in SOURCE_ROOT.')
        fonts.update(additional)
    seen, unseen = select_test_fonts(fonts, provenance)
    print('Training fonts:', provenance['train_fonts'], flush=True)
    print('Validation fonts (excluded from true unseen test):', provenance['validation_fonts'], flush=True)
    print('Truly unseen test fonts:', unseen, flush=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    size = state['image_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceStrokeModel(image_size=size, width=state['width'], pretrained=False).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    active_fonts = seen + unseen
    if not active_fonts:
        raise ValueError('No source fonts available for test')
    paths = sorted({path for font in active_fonts for slots in fonts[font].values()
                    for bank in slots.values() for candidates in bank.values() for path in candidates})
    loaded = read_components(paths, size, workers)
    stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
    local = Path(local_root) / ('run_' + stamp)
    output = checkpoint.parent / ('jamo_test_' + stamp)
    output.mkdir()
    plans = [('seen_standard', seen, 'standard'), ('seen_strong', seen, 'strong'),
             ('unseen_standard', unseen, 'standard'), ('unseen_strong', unseen, 'strong')]
    results, all_fingerprints = {}, set(provenance['fingerprints'])
    for offset, (name, selected, profile) in enumerate(plans):
        if not selected:
            results[name] = dict(status='skipped', reason='No eligible fonts; add a new FONT_unique_images folder.')
            continue
        manifest, generation = generate_test(fonts, loaded, selected, local / name,
                                            seed + offset, profile, samples_per_class, all_fingerprints, max_overlap)
        new_rows = read_manifest(manifest)
        all_fingerprints.update(row['image_sha256'] for row in new_rows)
        metrics = evaluate_test(model, manifest, device, size, batch_size)
        destination = output / name
        destination.mkdir()
        metrics.update(status='evaluated', generation=generation, fonts=selected,
                       reference_mode='same_font', checkpoint_epoch=state['epoch'])
        (destination / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        # Save reusable test dataset as one ZIP, with relative sample paths.
        packed = Path(shutil.make_archive(str(local / name), 'zip', manifest.parent))
        shutil.copyfile(packed, destination / 'dataset.zip')
        print(name, json.dumps(metrics['overall'], ensure_ascii=False), flush=True)
        if preview_count > 0:
            preview_reference(manifest, checkpoint, count=preview_count, split='test', save_dir=destination)
            worst = min(metrics['per_sample'], key=lambda row: row['metrics']['dice'])
            preview_reference(manifest, checkpoint, count=1, split='test', start=worst['sample_index'], save_dir=destination)
        results[name] = metrics
    summary = dict(checkpoint=str(checkpoint), checkpoint_epoch=state['epoch'],
                   training_archive=provenance['training_archive'], train_fonts=provenance['train_fonts'],
                   validation_fonts=provenance['validation_fonts'], seed=seed, tests=results,
                   note='All metrics exclude ambiguous overlap pixels. Seen-font tests reuse original banks with fresh deformations. Unseen-font tests require fonts absent from both training and validation.')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Test results saved:', output, flush=True)
    return summary, output
