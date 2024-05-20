import opencsp.common.lib.render_control.RenderControlLightPath as rclp


class RenderControlRayTrace():
    def __init__(self, light_path_control=rclp.default_path()) -> None:
        self.light_path_control = light_path_control


# Common Configurations

def init_current_lengths(init_len=1, current_len=1):
    return RenderControlRayTrace(rclp.RenderControlLightPath(init_length=init_len,
                                                             current_length=current_len))
