"""The OMX-L leader driver with its joint stream parked on a raw topic.

ROBOTIS's ``omx_l_leader_ai.launch.py`` makes the leader publish ``/leader/joint_trajectory`` -- the very topic the
follower's arm controller listens to, so the follower jumps to the leader's pose at full speed the moment the
leader comes up. This wrapper includes that launch under a ``SetRemap`` so that the leader publishes
``/leader/joint_trajectory_raw`` instead; ``python -m dlb.johnny.teleop_relay`` approaches the leader's pose slowly
and then forwards the stream with a velocity limit.

    ros2 launch dlb/johnny/launch/omx_l_leader_relay.launch.py port_name:=/dev/ttyACM1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("port_name", default_value="/dev/ttyACM1", description="the leader's USB port"),
        DeclareLaunchArgument(
            "raw_topic",
            default_value="/leader/joint_trajectory_raw",
            description="where the leader's joint stream goes (the relay reads it)",
        ),
        DeclareLaunchArgument("use_mock_hardware", default_value="false"),
    ]
    leader = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("open_manipulator_bringup"), "launch", "omx_l_leader_ai.launch.py"]
            )
        ),
        launch_arguments={
            "port_name": LaunchConfiguration("port_name"),
            "use_mock_hardware": LaunchConfiguration("use_mock_hardware"),
        }.items(),
    )
    return LaunchDescription(
        [
            *args,
            GroupAction(
                [SetRemap(src="/leader/joint_trajectory", dst=LaunchConfiguration("raw_topic")), leader]
            ),
        ]
    )
