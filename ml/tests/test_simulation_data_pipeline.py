"""Tests for the Gazebo-to-dataset pipeline orchestrator."""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path


ML_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ML_ROOT))

from run_simulation_data_pipeline import (  # noqa: E402, I100, I201
    ManagedProcess,
    Scenario,
    generate_lighting_world,
    load_scenarios,
    main,
    rosbag_command,
    scenario_command,
    session_count_report,
    update_session_tags,
    wait_for_log_marker,
)


def _scenario():
    return Scenario(
        scenario_id='near_normal_clear_straight',
        distance='near',
        lighting='normal',
        occlusion='none',
        duration=10.0,
        start_y=5.0,
        speed=0.8,
        travel_distance=15.0,
        lateral_amplitude=0.0,
        lateral_wavelength=10.0,
        occlusion_windows='',
    )


def test_checked_in_scenario_matrix_is_balanced_and_complete():
    """The default matrix should cover every required environment value."""
    scenarios = load_scenarios(
        ML_ROOT / 'config' / 'simulation_data_scenarios.yaml'
    )

    assert len(scenarios) == 10
    assert {item.distance for item in scenarios} == {
        'near', 'mid', 'far'
    }
    assert {item.lighting for item in scenarios} == {
        'normal', 'backlit', 'low'
    }
    assert {item.occlusion for item in scenarios} == {
        'none', 'partial'
    }


def test_generate_lighting_world_changes_scene_and_sun(tmp_path):
    """Generated low-light sessions must change pixels, not metadata alone."""
    source = tmp_path / 'source.sdf'
    source.write_text(
        '<sdf version="1.9"><world name="iris_runway">'
        '<scene><ambient>1 1 1 1</ambient>'
        '<background>1 1 1 1</background></scene>'
        '<light type="directional" name="sun">'
        '<diffuse>1 1 1 1</diffuse><specular>1 1 1 1</specular>'
        '<direction>0 0 -1</direction></light>'
        '</world></sdf>',
        encoding='utf-8',
    )
    output = tmp_path / 'low.sdf'

    settings = generate_lighting_world(source, output, 'low')

    world = element_tree.parse(output).getroot().find('world')
    assert world.find('scene/ambient').text == settings['ambient']
    assert (
        world.find("./light[@name='sun']/diffuse").text
        == settings['diffuse']
    )


def test_generated_world_embeds_grounded_moving_actor_trajectory(tmp_path):
    """Recorded scenarios must contain real pixel motion at ground height."""
    source = tmp_path / 'source.sdf'
    source.write_text(
        '<sdf version="1.9"><world name="iris_runway">'
        '<scene><ambient>1 1 1 1</ambient>'
        '<background>1 1 1 1</background></scene>'
        '<light type="directional" name="sun">'
        '<diffuse>1 1 1 1</diffuse><specular>1 1 1 1</specular>'
        '<direction>0 0 -1</direction></light>'
        '<actor name="runner"><pose>0 5 1.3 0 0 0</pose>'
        '<animation name="walking"/></actor>'
        '</world></sdf>',
        encoding='utf-8',
    )
    output = tmp_path / 'generated.sdf'

    settings = generate_lighting_world(
        source,
        output,
        'normal',
        scenario=_scenario(),
        actor_delay_start=15.0,
    )

    actor = element_tree.parse(output).getroot().find(
        "./world/actor[@name='runner']"
    )
    assert actor.findtext('pose') == '0 0 0 0 0 0'
    assert actor.findtext('script/delay_start') == '15.000000'
    waypoints = actor.findall('script/trajectory/waypoint')
    assert len(waypoints) > 2
    first_pose = waypoints[0].findtext('pose').split()
    last_pose = waypoints[-1].findtext('pose').split()
    assert float(first_pose[1]) == _scenario().start_y
    assert float(first_pose[2]) == 1.3
    assert float(last_pose[1]) > float(first_pose[1])
    assert settings['actor_trajectory']['planned_travel_metres'] > 0.0


def test_commands_include_scenario_parameters_and_mcap_metadata(tmp_path):
    """Commands should preserve trajectory controls and collection tags."""
    scenario = _scenario()
    dynamic = scenario_command(scenario)
    installed_dynamic = scenario_command(scenario, tmp_path / 'ros2_ws')
    arguments = argparse.Namespace(storage='mcap')
    record = rosbag_command(
        arguments,
        tmp_path / 'bag',
        scenario,
        ['/camera/image_raw'],
    )

    assert 'start_y:=5.0' in dynamic
    assert 'actor_height:=0.0' in dynamic
    assert 'lateral_amplitude:=0.0' in dynamic
    assert not any(value == 'occlusion_windows:=' for value in dynamic)
    assert installed_dynamic[0].endswith(
        'install/drone_simulation/lib/drone_simulation/'
        'dynamic_target_scenario'
    )
    assert installed_dynamic[1] == '--ros-args'
    assert '--storage' in record
    assert 'mcap' in record
    assert 'distance=near' in record


def test_managed_process_interrupts_and_reaps_child(tmp_path):
    """Shutdown must reap children rather than leave zombie processes."""
    process = ManagedProcess(
        'sleeper',
        [sys.executable, '-c', 'import time; time.sleep(60)'],
        tmp_path / 'sleeper.log',
        os.environ.copy(),
    )

    return_code = process.stop(0.2)

    assert return_code is not None
    assert process.poll() is not None


def test_log_marker_waits_for_semantic_child_readiness(tmp_path):
    """Recording time must begin after the dynamic target is initialized."""
    process = ManagedProcess(
        'ready_child',
        [
            sys.executable,
            '-c',
            'print("Dynamic target scenario ready:", flush=True); '
            'import time; time.sleep(60)',
        ],
        tmp_path / 'ready.log',
        os.environ.copy(),
    )
    try:
        elapsed = wait_for_log_marker(
            process, 'Dynamic target scenario ready:', 2.0
        )
        assert elapsed < 2.0
    finally:
        process.stop(0.2)


def test_session_tags_and_count_report_are_machine_readable(tmp_path):
    """Successful recordings should create ingestion-compatible tags."""
    tags = tmp_path / 'session_tags.json'
    report = {
        'status': 'recorded',
        'session_name': 'sim_run_001',
        'generated_world': '/tmp/world.sdf',
        'scenario': {
            **_scenario().__dict__,
        },
    }

    update_session_tags(tags, [report])

    payload = json.loads(tags.read_text('utf-8'))
    conditions = payload['sessions']['sim_run_001']['conditions']
    assert conditions == {
        'distance': 'near',
        'lighting': 'normal',
        'occlusion': 'none',
    }

    dataset = tmp_path / 'runner_raw'
    dataset.mkdir()
    (dataset / 'session_metadata.json').write_text(json.dumps({
        'sessions': {
            f'bag_{index:03d}': {
                'annotation_status': 'human_reviewed',
                'conditions': conditions,
            }
            for index in range(1, 4)
        }
    }), encoding='utf-8')
    count = session_count_report(dataset, minimum_sessions=3)
    assert count['status'] == 'passed'
    assert count['total_sessions'] == 3


def test_dry_run_writes_master_report_without_starting_ros(tmp_path):
    """Dry-run mode should be safe on CI systems without Gazebo."""
    report = tmp_path / 'pipeline_execution_report.json'

    return_code = main([
        '--dry-run',
        '--limit', '1',
        '--report', str(report),
        '--inbox', str(tmp_path / 'inbox'),
    ])

    payload = json.loads(report.read_text('utf-8'))
    assert return_code == 0
    assert payload['status'] == 'dry_run'
    assert len(payload['planned_recordings']) == 1
    assert not (tmp_path / 'inbox').exists()
