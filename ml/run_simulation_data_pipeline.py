#!/usr/bin/env python3
"""Record varied Gazebo sessions and run the Phase 1 dataset pipeline."""

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as element_tree
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DISTANCES = {'near', 'mid', 'far'}
LIGHTING_PROFILES = {'normal', 'backlit', 'low'}
OCCLUSIONS = {'none', 'partial'}
SESSION_ID_PATTERN = re.compile(r'^[a-z0-9][a-z0-9_-]*$')


def utc_now():
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def timestamp_id():
    """Return a filesystem-safe UTC identifier."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


@dataclass(frozen=True)
class Scenario:
    """One independent Gazebo recording scenario."""

    scenario_id: str
    distance: str
    lighting: str
    occlusion: str
    duration: float
    start_y: float
    speed: float
    travel_distance: float
    lateral_amplitude: float
    lateral_wavelength: float
    occlusion_windows: str


def parse_arguments(argv=None):
    """Parse recording and downstream pipeline controls."""
    project_root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario-config',
        type=Path,
        default=(
            Path(__file__).parent
            / 'config'
            / 'simulation_data_scenarios.yaml'
        ),
    )
    parser.add_argument('--scenario', action='append', default=[])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--duration', type=float)
    parser.add_argument(
        '--world',
        type=Path,
        default=(
            project_root
            / 'ros2_ws'
            / 'src'
            / 'drone_simulation'
            / 'worlds'
            / 'runner_tracking.sdf'
        ),
    )
    parser.add_argument(
        '--ros-workspace',
        type=Path,
        default=project_root / 'ros2_ws',
    )
    parser.add_argument(
        '--inbox',
        type=Path,
        default=project_root / 'data' / 'rosbag_inbox',
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=project_root / 'data' / 'runner_raw',
    )
    parser.add_argument(
        '--prepared-dir',
        type=Path,
        default=project_root / 'data' / 'runner_v1',
    )
    parser.add_argument(
        '--preannotate-model',
        default=str(project_root / 'yolov8n.pt'),
    )
    parser.add_argument('--skip-preannotation', action='store_true')
    parser.add_argument(
        '--record-topic',
        action='append',
        default=[],
        help='Repeat to record additional topics.',
    )
    parser.add_argument('--image-topic', default='/camera/image_raw')
    parser.add_argument(
        '--storage', choices=('mcap', 'sqlite3'), default='mcap'
    )
    parser.add_argument('--every-nth-frame', type=int, default=5)
    parser.add_argument('--minimum-sessions', type=int, default=10)
    parser.add_argument('--startup-timeout', type=float, default=90.0)
    parser.add_argument('--camera-warmup', type=float, default=15.0)
    parser.add_argument(
        '--actor-delay-start',
        type=float,
        default=2.0,
        help='Gazebo simulation seconds before the SDF runner path begins.',
    )
    parser.add_argument('--target-startup-timeout', type=float, default=30.0)
    parser.add_argument('--shutdown-timeout', type=float, default=10.0)
    parser.add_argument('--report', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--record-only', action='store_true')
    mode.add_argument('--process-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def _load_yaml(path):
    """Load one YAML mapping."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError('PyYAML is required for scenario files') from error
    payload = yaml.safe_load(path.read_text('utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')
    return payload


def load_scenarios(path, selected=(), limit=None, duration=None):
    """Load and validate the configured scenario matrix."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Scenario config not found: {path}')
    payload = _load_yaml(path)
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported scenario configuration schema')
    records = payload.get('sessions')
    if not isinstance(records, list) or not records:
        raise ValueError('Scenario config requires a non-empty sessions list')
    selected = set(selected)
    scenarios = []
    identifiers = set()
    for record in records:
        scenario_id = str(record.get('id', ''))
        if selected and scenario_id not in selected:
            continue
        if not SESSION_ID_PATTERN.fullmatch(scenario_id):
            raise ValueError(
                f'Invalid scenario ID {scenario_id!r}; use lowercase '
                'letters, digits, dash, and underscore'
            )
        if scenario_id in identifiers:
            raise ValueError(f'Duplicate scenario ID: {scenario_id}')
        identifiers.add(scenario_id)
        distance = str(record.get('distance', '')).lower()
        lighting = str(record.get('lighting', '')).lower()
        occlusion = str(record.get('occlusion', '')).lower()
        if distance not in DISTANCES:
            raise ValueError(f'Invalid distance for {scenario_id}: {distance}')
        if lighting not in LIGHTING_PROFILES:
            raise ValueError(f'Invalid lighting for {scenario_id}: {lighting}')
        if occlusion not in OCCLUSIONS:
            raise ValueError(
                f'Invalid occlusion for {scenario_id}: {occlusion}'
            )
        scenario_duration = (
            float(duration)
            if duration is not None
            else float(record.get('duration', 15.0))
        )
        if scenario_duration <= 0.0:
            raise ValueError(f'Duration must be positive for {scenario_id}')
        scenarios.append(Scenario(
            scenario_id=scenario_id,
            distance=distance,
            lighting=lighting,
            occlusion=occlusion,
            duration=scenario_duration,
            start_y=float(record['start_y']),
            speed=float(record['speed']),
            travel_distance=float(record['travel_distance']),
            lateral_amplitude=float(record.get('lateral_amplitude', 0.0)),
            lateral_wavelength=float(
                record.get('lateral_wavelength', 10.0)
            ),
            occlusion_windows=str(
                record.get('occlusion_windows', '')
            ),
        ))
    if selected - identifiers:
        missing = ', '.join(sorted(selected - identifiers))
        raise ValueError(f'Unknown selected scenarios: {missing}')
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive')
        scenarios = scenarios[:limit]
    if not scenarios:
        raise ValueError('No scenarios selected')
    return scenarios


def _configure_actor_trajectory(world, scenario, delay_start):
    """Embed a grounded, deterministic Gazebo actor trajectory in the world."""
    actor = world.find("./actor[@name='runner']")
    if actor is None:
        raise ValueError('World must contain an actor named runner')
    # Gazebo actor waypoint poses are relative to this trajectory origin. Keep
    # the origin neutral and put the complete world pose in each waypoint, as
    # in Gazebo Harmonic's official actor example.
    actor.find('pose').text = '0 0 0 0 0 0'
    existing = actor.find('script')
    if existing is not None:
        actor.remove(existing)
    script = element_tree.SubElement(actor, 'script')
    element_tree.SubElement(script, 'loop').text = 'false'
    element_tree.SubElement(script, 'delay_start').text = (
        f'{max(0.0, float(delay_start)):.6f}'
    )
    element_tree.SubElement(script, 'auto_start').text = 'true'
    trajectory = element_tree.SubElement(
        script,
        'trajectory',
        {'id': '0', 'type': 'walking', 'tension': '1.0'},
    )
    sample_period = 0.5
    sample_count = max(1, math.ceil(scenario.duration / sample_period))
    for index in range(sample_count + 1):
        elapsed = min(scenario.duration, index * sample_period)
        travelled = min(
            scenario.travel_distance, scenario.speed * elapsed
        )
        wavelength = max(0.5, scenario.lateral_wavelength)
        wave_number = 2.0 * math.pi / wavelength
        phase = wave_number * travelled
        lateral = scenario.lateral_amplitude * math.sin(phase)
        lateral_slope = (
            scenario.lateral_amplitude
            * wave_number
            * math.cos(phase)
        )
        path_heading = math.atan2(1.0, lateral_slope)
        actor_yaw = path_heading - math.pi
        waypoint = element_tree.SubElement(trajectory, 'waypoint')
        element_tree.SubElement(waypoint, 'time').text = f'{elapsed:.6f}'
        element_tree.SubElement(waypoint, 'pose').text = (
            f'{lateral:.6f} '
            f'{scenario.start_y + travelled:.6f} '
            f'1.300000 0 0 {actor_yaw:.6f}'
        )
    return {
        'delay_start_seconds': max(0.0, float(delay_start)),
        'duration_seconds': scenario.duration,
        'start_y_metres': scenario.start_y,
        'planned_travel_metres': min(
            scenario.travel_distance,
            scenario.speed * scenario.duration,
        ),
        'waypoint_count': sample_count + 1,
    }


def generate_lighting_world(
    source_world,
    output_world,
    profile,
    scenario=None,
    actor_delay_start=0.0,
):
    """Create a generated world with a controlled outdoor lighting profile."""
    source_world = source_world.expanduser().resolve()
    if not source_world.is_file():
        raise FileNotFoundError(f'Gazebo world not found: {source_world}')
    if profile not in LIGHTING_PROFILES:
        raise ValueError(f'Unsupported lighting profile: {profile}')
    tree = element_tree.parse(source_world)
    root = tree.getroot()
    world = root.find('world')
    if world is None:
        raise ValueError(f'No world element in {source_world}')
    scene = world.find('scene')
    sun = world.find("./light[@name='sun']")
    if scene is None or sun is None:
        raise ValueError('World must contain a scene and named sun light')
    settings = {
        'normal': {
            'ambient': '0.55 0.55 0.55 1',
            'background': '0.75 0.80 0.90 1',
            'diffuse': '0.80 0.80 0.80 1',
            'specular': '0.30 0.30 0.30 1',
            'direction': '-0.4 0.3 -0.85',
        },
        'backlit': {
            'ambient': '0.28 0.28 0.30 1',
            'background': '0.90 0.88 0.80 1',
            'diffuse': '1.00 0.92 0.75 1',
            'specular': '0.50 0.45 0.35 1',
            'direction': '0 -0.45 -0.89',
        },
        'low': {
            'ambient': '0.10 0.11 0.14 1',
            'background': '0.04 0.05 0.08 1',
            'diffuse': '0.22 0.24 0.30 1',
            'specular': '0.08 0.08 0.10 1',
            'direction': '-0.2 0.2 -0.95',
        },
    }[profile]
    for name in ('ambient', 'background'):
        element = scene.find(name)
        if element is None:
            element = element_tree.SubElement(scene, name)
        element.text = settings[name]
    for name in ('diffuse', 'specular', 'direction'):
        element = sun.find(name)
        if element is None:
            element = element_tree.SubElement(sun, name)
        element.text = settings[name]
    if scenario is not None:
        settings['actor_trajectory'] = _configure_actor_trajectory(
            world, scenario, actor_delay_start
        )
    output_world.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_world, encoding='utf-8', xml_declaration=True)
    return settings


class ManagedProcess:
    """One subprocess running in an isolated POSIX process group."""

    def __init__(self, name, command, log_path, environment):
        """Start a process and route its output to a session log."""
        self.name = name
        self.command = [str(value) for value in command]
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open('w', encoding='utf-8')
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            start_new_session=True,
        )

    @property
    def pid(self):
        """Return the child process ID."""
        return self.process.pid

    def poll(self):
        """Return the child exit status without blocking."""
        return self.process.poll()

    def stop(self, graceful_timeout):
        """Interrupt, terminate, then kill the entire child process group."""
        if self.process.poll() is None:
            for child_signal, timeout in (
                (signal.SIGINT, graceful_timeout),
                (signal.SIGTERM, 3.0),
                (signal.SIGKILL, 2.0),
            ):
                try:
                    os.killpg(self.process.pid, child_signal)
                except ProcessLookupError:
                    break
                try:
                    self.process.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if self.process.poll() is None:
            self.process.kill()
        try:
            return_code = self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return_code = None
        self._log.close()
        return return_code


class ProcessSupervisor:
    """Own and reap every process started for one simulation session."""

    def __init__(self, environment, shutdown_timeout):
        """Initialize process ownership for one recording session."""
        self.environment = environment
        self.shutdown_timeout = shutdown_timeout
        self.processes = []

    def start(self, name, command, log_path):
        """Start and register one managed process."""
        process = ManagedProcess(
            name, command, log_path, self.environment
        )
        self.processes.append(process)
        return process

    def stop(self, process):
        """Stop one registered process and remove it from active ownership."""
        return_code = process.stop(self.shutdown_timeout)
        if process in self.processes:
            self.processes.remove(process)
        return return_code

    def stop_all(self):
        """Stop every remaining process in reverse dependency order."""
        results = {}
        for process in reversed(self.processes[:]):
            results[process.name] = process.stop(self.shutdown_timeout)
            self.processes.remove(process)
        return results


def wait_for_log_marker(process, marker, timeout):
    """Wait until a child reports semantic readiness in its session log."""
    started = time.monotonic()
    deadline = started + max(0.1, float(timeout))
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f'{process.name} exited before readiness with {return_code}'
            )
        try:
            output = process.log_path.read_text(
                encoding='utf-8', errors='replace'
            )
        except OSError:
            output = ''
        if marker in output:
            return time.monotonic() - started
        time.sleep(0.1)
    raise RuntimeError(
        f'{process.name} did not report readiness within {timeout:.1f}s'
    )


def simulation_environment(arguments):
    """Build a bounded ROS/Gazebo subprocess environment."""
    environment = os.environ.copy()
    workspace = arguments.ros_workspace.expanduser().resolve()
    package_models = (
        workspace
        / 'install'
        / 'drone_simulation'
        / 'share'
        / 'drone_simulation'
        / 'models'
    )
    resource_paths = [
        str(package_models),
        str(Path.home() / 'ardupilot_gazebo' / 'models'),
    ]
    if environment.get('GZ_SIM_RESOURCE_PATH'):
        resource_paths.append(environment['GZ_SIM_RESOURCE_PATH'])
    plugin_paths = [
        str(Path.home() / 'ardupilot_gazebo' / 'build')
    ]
    if environment.get('GZ_SIM_SYSTEM_PLUGIN_PATH'):
        plugin_paths.append(environment['GZ_SIM_SYSTEM_PLUGIN_PATH'])
    environment.update({
        'GZ_SIM_RESOURCE_PATH': os.pathsep.join(resource_paths),
        'GZ_SIM_SYSTEM_PLUGIN_PATH': os.pathsep.join(plugin_paths),
        'ROS_AUTOMATIC_DISCOVERY_RANGE': 'LOCALHOST',
        'FASTDDS_BUILTIN_TRANSPORTS': 'UDPv4',
        'OMP_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '1',
    })
    return environment


def verify_runtime():
    """Require a sourced Linux ROS environment and collection executables."""
    if os.name != 'posix':
        raise RuntimeError(
            'Run this pipeline inside Ubuntu/WSL, not PowerShell'
        )
    missing = [
        command for command in ('ros2', 'gz')
        if shutil.which(command) is None
    ]
    if missing:
        raise RuntimeError(
            'Missing commands: '
            + ', '.join(missing)
            + '. Source ROS Jazzy and the workspace first.'
        )
    package = subprocess.run(
        ['ros2', 'pkg', 'prefix', 'drone_simulation'],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if package.returncode:
        raise RuntimeError(
            'drone_simulation is unavailable; build and source ros2_ws'
        )


def wait_for_topic(topic, timeout, environment, required_processes):
    """Require stable Gazebo and daemon-free ROS publisher discovery."""
    deadline = time.monotonic() + timeout
    last_output = ''
    while time.monotonic() < deadline:
        for process in required_processes:
            if process.poll() is not None:
                raise RuntimeError(
                    f'{process.name} exited early with {process.poll()}'
                )
        try:
            gazebo_topics = subprocess.run(
                ['gz', 'topic', '--list'],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
                env=environment,
            )
            available_topics = set(gazebo_topics.stdout.splitlines())
            if topic in available_topics:
                result = subprocess.run(
                    [
                        'ros2', 'topic', 'info',
                        '--no-daemon', '--spin-time', '2',
                        topic,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    env=environment,
                )
                last_output = result.stdout + result.stderr
                match = re.search(
                    r'Publisher count:\s*(\d+)', last_output
                )
                if match and int(match.group(1)) > 0:
                    return
            else:
                last_output = (
                    f'Gazebo topic not discovered; stderr='
                    f'{gazebo_topics.stderr}'
                )
        except subprocess.TimeoutExpired:
            last_output = 'Gazebo or ROS topic readiness probe timed out'
        time.sleep(1.0)
    raise RuntimeError(
        f'Topic {topic} was not ready within {timeout:.1f}s: {last_output}'
    )


def _ros_parameter(name, value):
    """Build one ROS parameter assignment argument."""
    if isinstance(value, bool):
        encoded = 'true' if value else 'false'
    else:
        encoded = str(value)
    return ['-p', f'{name}:={encoded}']


def scenario_command(scenario, ros_workspace=None):
    """Build the dynamic runner node command for one scenario."""
    if ros_workspace is None:
        executable = [
            'ros2', 'run', 'drone_simulation', 'dynamic_target_scenario',
        ]
    else:
        executable = [str(
            Path(ros_workspace).expanduser().resolve()
            / 'install'
            / 'drone_simulation'
            / 'lib'
            / 'drone_simulation'
            / 'dynamic_target_scenario'
        )]
    command = [*executable, '--ros-args']
    parameters = {
        'use_sim_time': True,
        'speed': scenario.speed,
        'distance': scenario.travel_distance,
        'start_y': scenario.start_y,
        # walk.dae uses a feet-level origin; keep the actor grounded in every
        # generated recording session.
        'actor_height': 0.0,
        'lateral_amplitude': scenario.lateral_amplitude,
        'lateral_wavelength': scenario.lateral_wavelength,
        'occlusion_windows': scenario.occlusion_windows,
    }
    for name, value in parameters.items():
        # An empty ROS parameter override (``name:=``) is invalid YAML and can
        # leave the process inside Ubuntu's exception reporter. The node's
        # declared default already represents an empty occlusion schedule.
        if value == '':
            continue
        command.extend(_ros_parameter(name, value))
    return command


def rosbag_command(arguments, bag_path, scenario, topics):
    """Build a deterministic rosbag2 record command."""
    command = [
        'ros2', 'bag', 'record',
        '--output', str(bag_path),
        '--storage', arguments.storage,
        '--disable-keyboard-controls',
        '--use-sim-time',
        '--topics',
        *topics,
        '--custom-data',
        f'scenario={scenario.scenario_id}',
        f'distance={scenario.distance}',
        f'lighting={scenario.lighting}',
        f'occlusion={scenario.occlusion}',
    ]
    return command


def validate_recorded_bag(bag_path, storage):
    """Check that rosbag2 finalized metadata and non-empty storage files."""
    extension = '.mcap' if storage == 'mcap' else '.db3'
    metadata = bag_path / 'metadata.yaml'
    storage_files = [
        path for path in bag_path.glob(f'*{extension}')
        if path.is_file() and path.stat().st_size > 0
    ]
    if not metadata.is_file() or not storage_files:
        raise RuntimeError(
            f'Incomplete rosbag output in {bag_path}; metadata or storage '
            'file is missing'
        )
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            'PyYAML is required to validate rosbag metadata'
        ) from error
    payload = yaml.safe_load(metadata.read_text('utf-8'))
    information = payload.get('rosbag2_bagfile_information', {})
    message_count = int(information.get('message_count', 0))
    if message_count < 1:
        raise RuntimeError(f'Rosbag contains zero messages: {bag_path}')
    return {
        'metadata': metadata.as_posix(),
        'message_count': message_count,
        'storage_files': [
            {
                'path': path.as_posix(),
                'bytes': path.stat().st_size,
            }
            for path in storage_files
        ],
    }


def record_scenario(arguments, scenario, run_id, index, environment):
    """Record one independent generated Gazebo scenario."""
    inbox = arguments.inbox.expanduser().resolve()
    session_name = (
        f'sim_{scenario.scenario_id}_{run_id}_{index:02d}'
    )
    bag_path = inbox / session_name
    logs = inbox / '_logs' / session_name
    world_path = inbox / '_generated_worlds' / f'{session_name}.sdf'
    if bag_path.exists():
        raise FileExistsError(f'Refusing to overwrite rosbag: {bag_path}')
    lighting_settings = generate_lighting_world(
        arguments.world,
        world_path,
        scenario.lighting,
        scenario=scenario,
        actor_delay_start=arguments.actor_delay_start,
    )
    topics = arguments.record_topic or [arguments.image_topic]
    if arguments.image_topic not in topics:
        topics = [arguments.image_topic, *topics]
    supervisor = ProcessSupervisor(
        environment, arguments.shutdown_timeout
    )
    started = time.monotonic()
    report = {
        'session_name': session_name,
        'scenario': asdict(scenario),
        'bag_path': bag_path.as_posix(),
        'generated_world': world_path.as_posix(),
        'lighting_settings': lighting_settings,
        'topics': topics,
        'started_at': utc_now(),
        'status': 'running',
        'commands': {},
        'processes': {},
    }
    recorder = None
    try:
        gazebo_command = [
            'gz', 'sim', '-s', '-r', '-v', '2', str(world_path)
        ]
        report['commands']['gazebo'] = gazebo_command
        gazebo = supervisor.start(
            'gazebo', gazebo_command, logs / 'gazebo.log'
        )
        report['processes']['gazebo_pid'] = gazebo.pid

        clock_bridge_command = [
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/iris_runway/set_pose@'
            'ros_gz_interfaces/srv/SetEntityPose',
        ]
        report['commands']['clock_service_bridge'] = clock_bridge_command
        clock_bridge = supervisor.start(
            'clock_service_bridge',
            clock_bridge_command,
            logs / 'clock_service_bridge.log',
        )
        report['processes']['clock_service_bridge_pid'] = clock_bridge.pid

        image_bridge_command = [
            'ros2', 'run', 'ros_gz_image', 'image_bridge',
            arguments.image_topic,
        ]
        report['commands']['image_bridge'] = image_bridge_command
        image_bridge = supervisor.start(
            'image_bridge',
            image_bridge_command,
            logs / 'image_bridge.log',
        )
        report['processes']['image_bridge_pid'] = image_bridge.pid
        record_command = rosbag_command(
            arguments, bag_path, scenario, topics
        )
        report['commands']['rosbag_record'] = record_command
        recorder = supervisor.start(
            'rosbag_record', record_command, logs / 'rosbag_record.log'
        )
        report['processes']['rosbag_record_pid'] = recorder.pid
        time.sleep(1.0)
        if recorder.poll() is not None:
            raise RuntimeError(
                f'ros2 bag record exited early with {recorder.poll()}'
            )
        warmup_deadline = time.monotonic() + arguments.camera_warmup
        while time.monotonic() < warmup_deadline:
            for process in (gazebo, image_bridge, recorder):
                if process.poll() is not None:
                    raise RuntimeError(
                        f'{process.name} exited during camera warm-up with '
                        f'{process.poll()}'
                    )
            time.sleep(
                min(
                    0.25,
                    max(0.0, warmup_deadline - time.monotonic()),
                )
            )

        deadline = time.monotonic() + scenario.duration
        while time.monotonic() < deadline:
            for process in (gazebo, image_bridge, recorder):
                if process.poll() is not None:
                    raise RuntimeError(
                        f'{process.name} exited during recording with '
                        f'{process.poll()}'
                    )
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

        report['processes']['rosbag_record_exit'] = supervisor.stop(recorder)
        recorder = None
        report['bag_validation'] = validate_recorded_bag(
            bag_path, arguments.storage
        )
        report['status'] = 'recorded'
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        report.update({'status': 'failed', 'error': str(error)})
    finally:
        if recorder is not None:
            report['processes']['rosbag_record_exit'] = supervisor.stop(
                recorder
            )
        report['processes']['shutdown_exit_codes'] = supervisor.stop_all()
        report['finished_at'] = utc_now()
        report['wall_duration_seconds'] = time.monotonic() - started
    return report


def _atomic_json(path, payload):
    """Write a JSON report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def update_session_tags(path, recording_reports):
    """Merge successful recording tags into the ingestion metadata file."""
    payload = {'schema_version': 1, 'sessions': {}}
    if path.is_file():
        payload = json.loads(path.read_text('utf-8'))
    sessions = payload.setdefault('sessions', {})
    for report in recording_reports:
        if report['status'] != 'recorded':
            continue
        scenario = report['scenario']
        sessions[report['session_name']] = {
            'conditions': {
                'distance': scenario['distance'],
                'lighting': scenario['lighting'],
                'occlusion': scenario['occlusion'],
            },
            'quality_flags': [],
            'simulation': {
                'trajectory': (
                    'straight'
                    if scenario['lateral_amplitude'] == 0.0
                    else 'curved'
                ),
                'start_y': scenario['start_y'],
                'speed': scenario['speed'],
                'lighting_world': report['generated_world'],
            },
        }
    _atomic_json(path, payload)
    return path


def run_stage(name, command, log_directory, environment, timeout=3600.0):
    """Run one finite pipeline stage and retain its complete output."""
    log_path = log_directory / f'{name}.log'
    started = time.monotonic()
    result = {
        'name': name,
        'command': [str(value) for value in command],
        'started_at': utc_now(),
        'log': log_path.as_posix(),
        'status': 'running',
    }
    try:
        completed = subprocess.run(
            [str(value) for value in command],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
        output = completed.stdout + completed.stderr
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding='utf-8')
        result.update({
            'return_code': completed.returncode,
            'status': 'passed' if completed.returncode == 0 else 'failed',
            'output_tail': output[-4000:],
        })
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or '') + (error.stderr or '')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding='utf-8')
        result.update({
            'return_code': None,
            'status': 'timed_out',
            'output_tail': output[-4000:],
        })
    result['finished_at'] = utc_now()
    result['duration_seconds'] = time.monotonic() - started
    return result


def session_count_report(dataset_root, minimum_sessions):
    """Count registered sessions and summarize their readiness."""
    metadata_path = dataset_root / 'session_metadata.json'
    if not metadata_path.is_file():
        return {
            'status': 'failed',
            'minimum_required': minimum_sessions,
            'total_sessions': 0,
            'error': f'Missing session registry: {metadata_path}',
        }
    metadata = json.loads(metadata_path.read_text('utf-8'))
    sessions = metadata.get('sessions', {})
    statuses = {}
    condition_counts = {}
    for record in sessions.values():
        annotation_status = record.get('annotation_status', 'unknown')
        statuses[annotation_status] = statuses.get(annotation_status, 0) + 1
        for name, value in record.get('conditions', {}).items():
            tag = f'{name}={value}'
            condition_counts[tag] = condition_counts.get(tag, 0) + 1
    total = len(sessions)
    return {
        'status': 'passed' if total >= minimum_sessions else 'failed',
        'minimum_required': minimum_sessions,
        'total_sessions': total,
        'annotation_status_counts': dict(sorted(statuses.items())),
        'condition_counts': dict(sorted(condition_counts.items())),
    }


def parse_ingested_sessions(report_path):
    """Return session IDs created by the latest ingestion stage."""
    if not report_path.is_file():
        return []
    report = json.loads(report_path.read_text('utf-8'))
    return [
        result['session_id']
        for result in report.get('results', [])
        if result.get('status') == 'ingested'
    ]


def blocked_stage(name, reason):
    """Create a structured record for a deliberately skipped stage."""
    return {
        'name': name,
        'status': 'blocked',
        'reason': reason,
        'started_at': None,
        'finished_at': None,
        'duration_seconds': 0.0,
    }


def run_downstream_pipeline(arguments, run_id, environment, report_dir):
    """Ingest, verify, pre-annotate, validate, and split all sessions."""
    project_root = Path(__file__).parents[1]
    scripts = project_root / 'ml' / 'scripts'
    dataset_root = arguments.dataset_root.expanduser().resolve()
    prepared_dir = arguments.prepared_dir.expanduser().resolve()
    inbox = arguments.inbox.expanduser().resolve()
    metadata_file = inbox / 'session_tags.json'
    logs = report_dir / 'processing_logs'
    steps = []

    ingestion_report = dataset_root / f'ingestion_{run_id}.json'
    ingestion = run_stage(
        'ingestion',
        [
            sys.executable,
            scripts / 'ingest_sessions.py',
            '--input-dir', inbox,
            '--dataset-root', dataset_root,
            '--metadata-file', metadata_file,
            '--every-nth-frame', str(arguments.every_nth_frame),
            '--non-interactive',
            '--environment', 'simulation',
            '--report', ingestion_report,
        ],
        logs,
        environment,
    )
    ingestion['report'] = ingestion_report.as_posix()
    steps.append(ingestion)

    counts = session_count_report(
        dataset_root, arguments.minimum_sessions
    )
    steps.append({'name': 'session_count_verification', **counts})
    new_sessions = parse_ingested_sessions(ingestion_report)

    if arguments.skip_preannotation:
        annotation = blocked_stage(
            'preannotation',
            'Disabled by --skip-preannotation; labels must be supplied '
            'manually.',
        )
    elif not new_sessions:
        annotation = {
            'name': 'preannotation',
            'status': 'skipped',
            'reason': 'No newly ingested sessions.',
        }
    elif ingestion['status'] == 'passed':
        review_manifest = (
            dataset_root / f'annotation_review_{run_id}.json'
        )
        command = [
            sys.executable,
            scripts / 'preannotate_yolo.py',
            '--dataset-root', dataset_root,
            '--model', arguments.preannotate_model,
            '--review-manifest', review_manifest,
        ]
        for session in new_sessions:
            command.extend(['--session', session])
        annotation = run_stage(
            'preannotation', command, logs, environment
        )
        annotation['review_manifest'] = review_manifest.as_posix()
    else:
        annotation = blocked_stage(
            'preannotation', 'Ingestion did not complete successfully.'
        )
    steps.append(annotation)

    validation_report = dataset_root / f'dataset_report_{run_id}.json'
    if annotation['status'] in ('passed', 'skipped'):
        validation = run_stage(
            'dataset_validation',
            [
                sys.executable,
                scripts / 'validate_yolo_dataset.py',
                '--dataset-root', dataset_root,
                '--class-count', '1',
                '--report', validation_report,
            ],
            logs,
            environment,
        )
        validation['report'] = validation_report.as_posix()
    else:
        validation = blocked_stage(
            'dataset_validation',
            'Annotations are not structurally complete.',
        )
    steps.append(validation)

    if counts['status'] != 'passed':
        split = blocked_stage(
            'dataset_split',
            f'Only {counts["total_sessions"]} sessions; '
            f'{arguments.minimum_sessions} required.',
        )
    elif validation['status'] != 'passed':
        split = blocked_stage(
            'dataset_split', 'Dataset validation did not pass.'
        )
    else:
        split = run_stage(
            'dataset_split',
            [
                sys.executable,
                scripts / 'prepare_dataset.py',
                '--dataset-root', dataset_root,
                '--session-metadata',
                dataset_root / 'session_metadata.json',
                '--output-dir', prepared_dir,
                '--val-fraction', '0.20',
                '--seed', '42',
                '--path-mode', 'relative',
            ],
            logs,
            environment,
        )
        if split['status'] == 'passed':
            split.update({
                'manifest': (
                    prepared_dir / 'splits_v1.json'
                ).as_posix(),
                'ultralytics_yaml': (
                    prepared_dir / 'runner_v1.yaml'
                ).as_posix(),
                'training_ready': (
                    counts['annotation_status_counts'].get(
                        'human_reviewed', 0
                    ) == counts['total_sessions']
                ),
                'note': (
                    'A provisional split may exist, but ml/train.py blocks '
                    'training until every session is human-reviewed.'
                ),
            })
    steps.append(split)
    if split['status'] == 'passed':
        split_validation = run_stage(
            'prepared_split_validation',
            [
                sys.executable,
                project_root / 'ml' / 'train.py',
                '--data', prepared_dir / 'runner_v1.yaml',
                '--dry-run',
                '--allow-unreviewed',
                '--no-custom-augmentations',
            ],
            logs,
            environment,
        )
    else:
        split_validation = blocked_stage(
            'prepared_split_validation',
            'The prepared dataset split was not generated.',
        )
    steps.append(split_validation)
    return {
        'status': (
            'passed'
            if all(
                step.get('status') in ('passed', 'skipped')
                for step in steps
            )
            else 'incomplete'
        ),
        'new_session_ids': new_sessions,
        'steps': steps,
    }


def planned_commands(arguments, scenarios, run_id):
    """Return a dry-run preview without requiring ROS or creating files."""
    topics = arguments.record_topic or [arguments.image_topic]
    plans = []
    for index, scenario in enumerate(scenarios, start=1):
        session_name = (
            f'sim_{scenario.scenario_id}_{run_id}_{index:02d}'
        )
        bag_path = arguments.inbox.expanduser().resolve() / session_name
        plans.append({
            'session_name': session_name,
            'scenario': asdict(scenario),
            'dynamic_target_command': scenario_command(
                scenario, arguments.ros_workspace
            ),
            'rosbag_command': rosbag_command(
                arguments, bag_path, scenario, topics
            ),
        })
    return plans


def execute(arguments):
    """Execute the selected recording and processing modes."""
    if arguments.every_nth_frame < 1:
        raise ValueError('--every-nth-frame must be at least 1')
    if arguments.minimum_sessions < 1:
        raise ValueError('--minimum-sessions must be at least 1')
    run_id = timestamp_id()
    scenarios = load_scenarios(
        arguments.scenario_config,
        arguments.scenario,
        arguments.limit,
        arguments.duration,
    )
    project_root = Path(__file__).parents[1]
    report_path = arguments.report
    if report_path is None:
        report_path = (
            project_root
            / 'artifacts'
            / 'simulation_data_pipeline'
            / run_id
            / 'pipeline_execution_report.json'
        )
    report_path = report_path.expanduser().resolve()
    report = {
        'schema_version': 1,
        'run_id': run_id,
        'status': 'running',
        'started_at': utc_now(),
        'configuration': {
            'scenario_config': (
                arguments.scenario_config.expanduser().resolve().as_posix()
            ),
            'world': arguments.world.expanduser().resolve().as_posix(),
            'inbox': arguments.inbox.expanduser().resolve().as_posix(),
            'dataset_root': (
                arguments.dataset_root.expanduser().resolve().as_posix()
            ),
            'prepared_dir': (
                arguments.prepared_dir.expanduser().resolve().as_posix()
            ),
            'storage': arguments.storage,
            'every_nth_frame': arguments.every_nth_frame,
            'minimum_sessions': arguments.minimum_sessions,
            'actor_delay_start': arguments.actor_delay_start,
        },
        'recordings': [],
        'processing': None,
    }
    if arguments.dry_run:
        report.update({
            'status': 'dry_run',
            'planned_recordings': planned_commands(
                arguments, scenarios, run_id
            ),
            'finished_at': utc_now(),
        })
        _atomic_json(report_path, report)
        return report_path, report

    try:
        verify_runtime()
        environment = simulation_environment(arguments)
        if not arguments.process_only:
            for index, scenario in enumerate(scenarios, start=1):
                recording = record_scenario(
                    arguments, scenario, run_id, index, environment
                )
                report['recordings'].append(recording)
                _atomic_json(report_path, report)
            tags_path = (
                arguments.inbox.expanduser().resolve()
                / 'session_tags.json'
            )
            update_session_tags(tags_path, report['recordings'])
            report['session_tags'] = tags_path.as_posix()
        if not arguments.record_only:
            report['processing'] = run_downstream_pipeline(
                arguments,
                run_id,
                environment,
                report_path.parent,
            )
        recordings_passed = all(
            item['status'] == 'recorded'
            for item in report['recordings']
        )
        processing_passed = (
            arguments.record_only
            or report['processing']['status'] == 'passed'
        )
        report['status'] = (
            'passed'
            if recordings_passed and processing_passed
            else 'incomplete'
        )
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        report['error'] = 'Interrupted by operator'
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        report['status'] = 'failed'
        report['error'] = str(error)
    finally:
        report['finished_at'] = utc_now()
        _atomic_json(report_path, report)
    return report_path, report


def main(argv=None):
    """Run the simulation-data pipeline and emit one master report."""
    arguments = parse_arguments(argv)
    try:
        report_path, report = execute(arguments)
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        RuntimeError,
        ValueError,
        element_tree.ParseError,
    ) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    print(f'Pipeline status: {report["status"]}')
    print(f'Master report: {report_path}')
    return 0 if report['status'] in ('passed', 'dry_run') else 1


if __name__ == '__main__':
    raise SystemExit(main())
