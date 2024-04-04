import opencsp.common.lib.camera.ImageAcquisitionAbstract as iaa
import opencsp.common.lib.deflectometry.ImageProjection as ip
import opencsp.app.sofast.lib.SystemSofastFringe as ssf


class SofastServiceCallback:
    def on_service_set(self):
        pass

    def on_service_unset(self):
        pass

    def on_image_projection_set(self, image_projection: ip.ImageProjection):
        self.on_service_set()

    def on_image_projection_unset(self, image_projection: ip.ImageProjection):
        self.on_service_unset()

    def on_image_acquisition_set(self, image_acquisition: iaa.ImageAcquisitionAbstract):
        self.on_service_set()

    def on_image_acquisition_unset(self, image_acquisition: iaa.ImageAcquisitionAbstract):
        self.on_service_unset()

    def on_system_set(self, system: ssf.SystemSofastFringe):
        self.on_service_set()

    def on_system_unset(self, system: ssf.SystemSofastFringe):
        self.on_service_unset()
