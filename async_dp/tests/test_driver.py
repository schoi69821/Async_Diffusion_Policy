from src.drivers.robot_driver import RobotDriver
def test_dummy_mode():
    d = RobotDriver(mode='dummy')
    assert len(d.get_qpos()) == 14
    d.close()