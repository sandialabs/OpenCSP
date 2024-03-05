import opencsp.common.lib.render_control.RenderControlLightPath as rclp 


class RenderControlRayTrace():
    def __init__(self, light_path_control: float = rclp.default_path()) -> None:
        self.light_path_control = light_path_control

# Common Configurations
