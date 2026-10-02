"""Final cohort reports must be complete, portable and honest about missing grades."""
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from benchmark.measurement import suite_hash, write_json


@pytest.fixture
def publisher():
    script = Path(__file__).resolve().parents[3] / 'tools/publish_asset_cohorts.py'
    sys.path.insert(0, str(script.parent))
    try:
        spec = importlib.util.spec_from_file_location('asset_cohort_report', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script.parent))


@pytest.fixture
def experiment(tmp_path):
    cohorts = tmp_path / 'cohorts'
    root = tmp_path / 'live'
    config = {'asset_class': 'Chiller', 'judge': 'claude-code/claude-fable-5-1', 'targets': []}
    all_scenarios = []
    for cohort in ('existing', 'synthetic'):
        suite = cohorts / cohort
        rows = [{'id': str(i), 'type': 'iot', 'text': f'{cohort} chiller question {i}',
                 'entity': 'Chiller', 'characteristic_form': 'Read and verify actual sensor values.'}
                for i in (1, 2)]
        all_scenarios.extend(rows)
        write_json(suite / 'run.json', {'status': 'complete', 'config': {'num_scenarios': 2,
                   'num_negative_scenarios': 0}, 'scenario_count': 2, 'negative_count': 0})
        write_json(suite / 'scenarios.json', rows)
        config['targets'].append({'name': f'{cohort}-executor', 'cohort': cohort,
                                  'model_key': 'gpt-6-1-sol', 'model_id': 'gpt-6.1-sol',
                                  'agent': 'codex', 'suite': str(suite)})
    write_json(cohorts / 'cohorts.json', {'existing_count': 6, 'selected_count': 2,
                                       'asset_class': 'Chiller', 'inventory': None})
    write_json(cohorts / 'all_existing.json', all_scenarios)
    write_json(cohorts / 'generation-config.json', {'model_id': 'claude-opus-5-5', 'mode': 'open'})
    (cohorts / 'synthetic/logs').mkdir(parents=True)
    (cohorts / 'synthetic/logs/prompt.md').write_text('Reference real registered assets.')
    snapshot = 'shared-snapshot'
    repetitions = []
    for index in (1, 2, 3):
        repetition = {'index': index, 'root': str(root / f'repetition-{index}')}
        repetitions.append(repetition)
        for target in config['targets']:
            suite = Path(target['suite'])
            folder = Path(repetition['root']) / target['name']
            settings = {'model': target['model_id'], 'agent': target['agent'],
                        'suite_sha256': suite_hash([suite / 'scenarios.json']),
                        'judge_model': config['judge'], 'repetition_index': index,
                        'invocation_retry_policy': {'max_attempts': 3},
                        'database_policy': {'snapshot_sha256': snapshot}}
            write_json(folder / 'settings.json', settings)
            write_json(folder / 'target.json', target)
            write_json(folder / 'environment.json', {'snapshot_sha256': snapshot})
            for sid in ('1', '2'):
                run_id = f"{target['name']}-{sid}"
                trace = folder / 'traces' / f'{run_id}.jsonl'
                trace.parent.mkdir(parents=True, exist_ok=True)
                trace.write_text(json.dumps({'kind': 'final_answer', 'answer': 'The observed value is 4.'}) + '\n')
                passed = target['cohort'] == 'synthetic' or index == 1 or (index == 2 and sid == '2')
                judge_trace = folder / 'traces' / f'{run_id}.judge.jsonl'
                judge_trace.write_text(json.dumps({'kind': 'judge_result', 'payload': {
                    'session_id': f"fresh-{index}-{run_id}"}}) + '\n')
                record = {'run_id': run_id, 'scenario_id': sid, 'execution_index': int(sid),
                          'attempt': 1, 'status': 'completed', 'settings': settings,
                          'trace_file': str(trace), 'execution_duration_ms': 10 * index,
                          'metrics': {'tool_call_count': 2, 'input_tokens': 100},
                          'grading': {'status': 'completed', 'model': config['judge'],
                                      'separate_session': True, 'trace_file': str(judge_trace),
                                      'duration_ms': 5,
                                      'attempts': [{'status': 'completed', 'trace_file': str(judge_trace)}],
                                      'result': {'score': {'passed': passed, 'score': float(passed),
                                          'rationale': 'Observed evidence supports this outcome.',
                                          'details': {'task_completion': passed,
                                                      'data_retrieval_accuracy': True,
                                                      'generalized_result_verification': passed,
                                                      'agent_sequence_correct': True,
                                                      'clarity_and_justification': True,
                                                      'hallucinations': False}}}}}
                if target['cohort'] == 'existing' and index == 3 and sid == '2':
                    # Terminal invocation failure has no successful judge or trace.
                    judge_trace.unlink()
                    trace.unlink()
                    record.update(status='failed', grading=None, error={'type': 'RuntimeError'}, attempt=3)
                    for attempt in (1, 2):
                        prior = deepcopy(record)
                        prior['attempt'] = attempt
                        write_json(folder / 'measurements' / f'{run_id}.attempt-{attempt}.json', prior)
                    (folder / f'{run_id}.log').write_text('Startup failed before trace capture.')
                else:
                    answer = ('Literal </script><script>evil()</script> should stay text.'
                              if sid == '1' else 'The observed value is 4.')
                    write_json(folder / 'trajectories' / f'{run_id}.json', {'answer': answer})
                write_json(folder / 'measurements' / f'{run_id}.json', record)
    result = {'status': 'completed', 'k': 3, 'config': config,
              'snapshot_sha256': snapshot, 'repetitions': repetitions,
              'suite_hashes': {s['name']: suite_hash([Path(s['suite']) / 'scenarios.json'])
                               for s in config['targets']}}
    path = root / 'experiment.json'
    write_json(path, result)
    return path, cohorts, result


def test_terminal_failure_preserves_strict_denominator_and_missing_criteria(publisher, experiment):
    _, _, data = experiment
    groups = publisher.collect(data)
    existing = groups['existing-executor']
    assert existing['means']['pass_rate'] == {'mean': .5, 'sd': .5, 'observed_repetitions': 3}
    assert existing['median_pass_rate'] == .5
    assert existing['pooled_cases']['graded'] == 5
    assert existing['pooled_cases']['assigned_cases'] == 6
    assert existing['pooled_cases']['execution_failed_cases'] == 1
    assert existing['pooled_attempts']['attempted'] == 8
    failed = existing['per_scenario']['2']['outcomes'][-1]
    assert failed['passed'] is False and failed['graded'] is False and failed['score'] is None
    assert existing['pooled_cases']['rubric_success_rates']['data_retrieval_accuracy'] == {
        'success_rate': 1, 'observed': 5}
    paper, hard = publisher.report_tables(data['config'], groups)
    assert '100.0% ± 0.0 pp' in paper and '5 / 6' in paper
    assert '50.0% ± 50.0 pp' in hard
    existing['means']['median_execution_ms'] = {'mean': 34000, 'sd': 1000}
    resources = publisher.resource_table(data['config'], groups)
    assert '34.0 ± 1.0' in resources and '2/2/1' in resources
    assert '200.0 ± 0.0' in resources


@pytest.mark.parametrize('change', ['missing_repeat', 'duplicate_repeat', 'missing_cohort', 'different_model'])
def test_false_k3_or_unmatched_comparisons_are_rejected(publisher, experiment, change):
    _, _, original = experiment
    data = deepcopy(original)
    if change == 'missing_repeat':
        data['repetitions'].pop()
    elif change == 'duplicate_repeat':
        data['repetitions'][-1]['index'] = 2
    elif change == 'missing_cohort':
        data['config']['targets'].pop()
    else:
        data['config']['targets'][-1]['model_id'] = 'other-model'
    with pytest.raises(ValueError):
        publisher.collect(data)


@pytest.mark.parametrize('change', ['suite_bytes', 'snapshot', 'judge', 'repeat', 'duplicate_attempt', 'missing_judgment'])
def test_inconsistent_or_incomplete_evidence_cannot_be_averaged(publisher, experiment, change):
    _, _, data = experiment
    spec = data['config']['targets'][0]
    target = Path(data['repetitions'][0]['root']) / spec['name']
    path = next((target / 'measurements').glob('*.json'))
    record = json.loads(path.read_text())
    if change == 'suite_bytes':
        suite = Path(spec['suite']) / 'scenarios.json'
        suite.write_text(suite.read_text() + '\n')
    elif change == 'snapshot':
        record['settings']['database_policy']['snapshot_sha256'] = 'wrong'
    elif change == 'judge':
        record['settings']['judge_model'] = 'wrong'
    elif change == 'repeat':
        record['settings']['repetition_index'] = 3
    elif change == 'duplicate_attempt':
        write_json(path.parent / 'duplicate.json', record)
    else:
        record['grading'] = None
    write_json(path, record)
    with pytest.raises(ValueError):
        publisher.collect(data)


def test_matched_cohorts_cannot_configure_different_reasoning_effort(publisher, experiment):
    _, _, data = experiment
    data['config']['targets'][0]['reasoning_effort'] = 'low'
    data['config']['targets'][1]['reasoning_effort'] = 'high'
    with pytest.raises(ValueError, match='configured reasoning effort'):
        publisher.collect(data)


def test_consistent_measurements_still_must_match_configured_reasoning(publisher, experiment):
    _, _, data = experiment
    for spec in data['config']['targets']:
        spec['reasoning_effort'] = 'low'
        for rep in data['repetitions']:
            for path in (Path(rep['root']) / spec['name'] / 'measurements').glob('*.json'):
                record = json.loads(path.read_text())
                record['settings']['reasoning_effort'] = 'high'
                write_json(path, record)
    with pytest.raises(ValueError, match='reasoning'):
        publisher.collect(data)


@pytest.mark.parametrize('field,value', [
    ('provider', 'another-provider'), ('harness', 'different-harness'),
    ('reasoning_effort', 'high'), ('cli_version', 'upgraded-cli'),
    ('sdk_version', 'upgraded-sdk'), ('timeout_seconds', 300),
    ('max_turns', 10), ('grading_settings', {'trajectory_character_limit': 100}),
    ('invocation_retry_policy', {'max_attempts': 1}),
])
def test_matched_cohorts_must_use_the_same_actual_runtime(publisher, experiment, field, value):
    _, _, data = experiment
    spec = data['config']['targets'][1]
    # Every synthetic repetition agrees internally, so this specifically tests
    # the cross-cohort check rather than the older within-target aggregation.
    for rep in data['repetitions']:
        for path in (Path(rep['root']) / spec['name'] / 'measurements').glob('*.json'):
            record = json.loads(path.read_text())
            record['settings'][field] = value
            write_json(path, record)
    with pytest.raises(ValueError, match='actual runtime settings'):
        publisher.collect(data)


def test_unconfigured_reasoning_and_legacy_repetition_metadata_remain_accepted(publisher, experiment):
    _, _, data = experiment
    for spec in data['config']['targets']:
        spec['reasoning_effort'] = None
        for rep in data['repetitions']:
            for path in (Path(rep['root']) / spec['name'] / 'measurements').glob('*.json'):
                record = json.loads(path.read_text())
                record['settings']['reasoning_effort'] = 'low'
                record['settings']['repetition_index'] = None
                write_json(path, record)
    assert publisher.collect(data)['existing-executor']['complete_repetitions'] == 3


def test_evidence_is_portable_and_missing_failed_trace_is_explicit(publisher, experiment, tmp_path):
    _, _, data = experiment
    dest = tmp_path / 'report'
    judged, all_judgments = publisher.export_evidence(data, dest, lambda text: text)
    assert judged == all_judgments == 11
    for path in dest.glob('repetition-*/*/measurements/*.json'):
        record = json.loads(path.read_text())
        if record['status'] == 'failed':
            assert record['trace_file'] is None
            assert record['trace_file_available'] is False
            assert record['missing_trace_file'].endswith('.jsonl')
            assert record['grading'] is None
        else:
            refs = [record['trace_file'], record['grading']['trace_file'],
                    record['grading']['attempts'][0]['trace_file']]
            for ref in refs:
                assert not Path(ref).is_absolute()
                assert gzip.decompress((dest / ref).read_bytes())
    assert list(dest.glob('repetition-*/*/*.log.txt'))
    assert not list(dest.glob('repetition-*/*/*.log'))


@pytest.mark.parametrize('change', ['missing_success_trace', 'reused_session', 'wrong_judge', 'shared_session'])
def test_successful_judge_evidence_must_exist_and_be_independent(publisher, experiment, tmp_path, change):
    _, _, data = experiment
    root = Path(data['repetitions'][0]['root'])
    records = [p for spec in data['config']['targets'] for p in
               (root / spec['name'] / 'measurements').glob('*.json')]
    first, second = (json.loads(p.read_text()) for p in records[:2])
    if change == 'missing_success_trace':
        Path(first['grading']['trace_file']).unlink()
    elif change == 'reused_session':
        Path(second['grading']['trace_file']).write_text(Path(first['grading']['trace_file']).read_text())
    elif change == 'wrong_judge':
        first['grading']['model'] = 'wrong'
    else:
        first['grading']['separate_session'] = False
    write_json(records[0], first)
    with pytest.raises(ValueError):
        publisher.export_evidence(data, tmp_path / 'report', lambda text: text)


def test_complete_report_rebuilds_from_its_own_evidence_and_keeps_case_links(publisher, experiment, tmp_path, monkeypatch):
    path, cohorts, data = experiment
    dest = tmp_path / 'report'
    # Graph rendering is orthogonal; the real graph function is smoke-tested below.
    def graph_stub(output, *_):
        folder = output / 'graphs'
        folder.mkdir(exist_ok=True)
        for name in ('criterion-averages', 'pass-rate'):
            (folder / f'{name}.png').write_bytes(b'image-placeholder')
    monkeypatch.setattr(publisher, 'graphs', graph_stub)
    publisher.publish(path, cohorts, dest, tmp_path / 'missing.env')
    manifest = json.loads((dest / 'manifest.json').read_text())
    assert manifest['assigned_trials'] == 12 and manifest['graded_results'] == 11
    assert manifest['execution_failed_cases'] == 1 and manifest['model_count'] == 1
    published = publisher.load_experiment(dest / 'experiment.json')
    assert publisher.collect(published) == publisher.collect(data)
    cases = json.loads((dest / 'cases.json').read_text())
    assert len(cases) == 12
    failed = next(case for case in cases if case['status'] == 'execution_failed')
    assert failed['score'] is None and failed['rubric'] is None and failed['passed'] is False
    for case in cases:
        for key in ('measurement', 'trajectory', 'execution_trace', 'judge_trace'):
            assert case[key] is None or (dest / case[key]).is_file()
    with (dest / 'cases.csv').open(newline='') as file:
        row = next(row for row in csv.DictReader(file) if row['status'] == 'execution_failed')
        assert row['passed'] == 'False' and row['task_completion'] == '' and row['score'] == ''
    assert not list(dest.rglob('*.html'))
    assert 'Literal </script>' in cases[0]['answer']
    assert 'graphs/criterion-averages.png' in (dest / 'README.md').read_text()
    assert 'graphs/pass-rate.png' in (dest / 'README.md').read_text()
    for line in (dest / 'checksums.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        assert hashlib.sha256((dest / relative).read_bytes()).hexdigest() == digest
    original_summary = (dest / 'summary.json').read_bytes()
    publisher.publish(dest / 'experiment.json', dest, dest, tmp_path / 'missing.env')
    assert (dest / 'summary.json').read_bytes() == original_summary
    assert list(dest.glob('repetition-*/*/*.log.txt'))
    assert not list(dest.glob('repetition-*/*/*.log.txt.txt'))
    assert (dest / 'generation-evidence/prompt.md').read_text() == 'Reference real registered assets.'
    assert '--cohorts <report-directory>' in (dest / 'README.md').read_text()


def test_graphs_accept_arbitrary_target_names_and_missing_criterion(publisher, experiment, tmp_path):
    pytest.importorskip('matplotlib')
    _, _, data = experiment
    groups = publisher.collect(data)
    for group in groups.values():
        group['means']['rubric_success_rates'].pop('generalized_result_verification')
    publisher.graphs(tmp_path, data['config'], groups)
    for name in ('criterion-averages', 'pass-rate'):
        assert (tmp_path / f'graphs/{name}.png').stat().st_size > 1000
        assert (tmp_path / f'graphs/{name}.svg').stat().st_size > 1000


def test_paper_plot_preserves_missing_rates_and_sample_sd_geometry():
    pytest.importorskip('matplotlib')
    from benchmark.paper_plots import criterion_figure, plotting
    missing = {'model': 'Unavailable', 'metrics': {key: {'mean': None, 'sd': None}
                                                  for key in ('task_completion', 'data_retrieval_accuracy', 'generalized_result_verification')}}
    observed = {'model': 'Observed', 'metrics': {key: {'mean': .5, 'sd': .1}
                                                for key in missing['metrics']}}
    fig = criterion_figure([('Existing', [missing, observed])], ['Unavailable', 'Observed'])
    axis = fig.axes[0]
    assert len(axis.patches) == 3  # Missing rates are not drawn as zeros.
    assert all(patch.get_height() == 50 for patch in axis.patches)
    assert all(text.get_rotation() == 90 for text in axis.texts)
    vertical_segments = [segment for collection in axis.collections for segment in collection.get_segments()]
    assert len(vertical_segments) == 3
    assert all(segment[0][0] == segment[1][0] and list(segment[:, 1]) == [40, 60]
               for segment in vertical_segments)
    plotting().close(fig)


@pytest.fixture
def progress_publisher(publisher):
    script = publisher.ROOT / 'tools/publish_asset_cohort_progress.py'
    sys.path.insert(0, str(script.parent))
    try:
        spec = importlib.util.spec_from_file_location('asset_cohort_progress_report', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.final = publisher
        return module
    finally:
        sys.path.remove(str(script.parent))


def interrupt_synthetic_r3(data):
    data['status'] = 'running'
    spec = data['config']['targets'][1]
    target = Path(data['repetitions'][-1]['root']) / spec['name']
    for path in (target / 'measurements').glob('*.json'):
        record = json.loads(path.read_text())
        if record['scenario_id'] == '1':
            record.update(status='failed', attempt=3, grading=None,
                          agent_error={'message': "You've hit your session limit · resets 5:30pm (America/New_York)"})
        else:
            record['grading'] = {'status': 'failed', 'error': {
                'message': "judge backend error: You've hit your session limit · resets 5:30pm"}}
        write_json(path, record)


def test_progress_uses_only_common_finished_repeats_without_quota_failure_rates(progress_publisher, experiment):
    _, _, data = experiment
    interrupt_synthetic_r3(data)
    matched, groups = progress_publisher.collect_progress(data)
    assert matched == [1, 2]
    assert all(group['k'] == 2 for group in groups.values())
    assert groups['existing-executor']['means']['pass_rate']['mean'] == .75
    assert groups['synthetic-executor']['means']['pass_rate']['mean'] == 1
    assert all(group['pooled_cases']['assigned_cases'] == 4 for group in groups.values())


def test_progress_keeps_quota_and_unjudged_scores_missing(progress_publisher, experiment, tmp_path):
    _, _, data = experiment
    interrupt_synthetic_r3(data)
    dest = tmp_path / 'progress'
    progress_publisher.final.export_evidence(data, dest, lambda text: text)
    portable = deepcopy(data)
    for rep in portable['repetitions']:
        rep['root'] = str(dest / f"repetition-{rep['index']}")
    cases, progress = progress_publisher.progress_cases(portable, dest)
    assert len(cases) == 12
    quota = [case for case in cases if case['status'] == 'quota_blocked']
    assert {case['quota_phase'] for case in quota} == {'execution', 'judge'}
    assert all(case['passed'] is None and case['score'] is None and case['rubric'] is None for case in quota)
    actual = next(case for case in cases if case['status'] == 'execution_failed')
    assert actual['passed'] is False and actual['score'] is None
    source = Path(portable['repetitions'][-1]['root']) / data['config']['targets'][1]['name'] / 'measurements'
    path = next(path for path in source.glob('*.json') if json.loads(path.read_text())['scenario_id'] == '2')
    record = json.loads(path.read_text())
    record['grading'] = None
    write_json(path, record)
    cases, _ = progress_publisher.progress_cases(portable, dest)
    pending = next(case for case in cases if case['status'] == 'pending_judge')
    assert pending['passed'] is None and pending['score'] is None and pending['rubric'] is None


def test_progress_cannot_replace_a_completed_final_k3_report(progress_publisher, experiment):
    _, _, data = experiment
    with pytest.raises(ValueError, match='final report'):
        progress_publisher.collect_progress(data)


def test_progress_publication_is_explicit_and_rebuildable(progress_publisher, publisher, experiment, tmp_path, monkeypatch):
    path, cohorts, data = experiment
    interrupt_synthetic_r3(data)
    write_json(path, data)
    dest = tmp_path / 'progress'
    def graph_stub(output, *_):
        graph_dir = output / 'graphs'
        graph_dir.mkdir(exist_ok=True)
        for name in ('criterion-averages', 'pass-rate'):
            (graph_dir / f'{name}.png').write_bytes(b'image-placeholder')
    monkeypatch.setattr(publisher, 'graphs', graph_stub)
    progress_publisher.publish(path, cohorts, dest, tmp_path / 'missing.env')
    manifest = json.loads((dest / 'manifest.json').read_text())
    assert manifest['status'] == 'incomplete'
    assert manifest['requested_k'] == 3 and manifest['observed_k'] == 2
    assert manifest['matched_repetitions'] == [1, 2]
    assert manifest['quota_blocked_cases'] == 2
    readme = (dest / 'README.md').read_text()
    assert '**In progress:**' in readme and '2 completed repetitions' in readme
    assert '| Model | Existing (%) | Synthetic (%) |' in readme
    assert '| R1 | R2 | R3 |' not in readme
    assert not list(dest.rglob('*.html'))
    with pytest.raises(ValueError, match='must complete'):
        publisher.collect(publisher.load_experiment(dest / 'experiment.json'))
    summary = json.loads((dest / 'summary.json').read_text())
    progress_publisher.publish(dest / 'experiment.json', dest, dest, tmp_path / 'missing.env')
    assert json.loads((dest / 'summary.json').read_text())['groups'] == summary['groups']
