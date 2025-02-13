from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import KeyEvent
import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.camera.image_processing import highlight_saturation
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
import opencsp.app.sofast.lib.sofast_common_functions as scf
import opencsp.common.lib.render.lib.AbstractPlotHandler as aph


class LiveView(aph.AbstractPlotHandler):
    def __init__(
        self, image_acquisition: ImageAcquisitionAbstract = None, update_ms: int = 20, highlight_saturation: bool = True
    ):
        """
        Shows live stream from a camera. Escape key closes window.

        Parameters
        ----------
        image_acquisition : ImageAcquisitionAbstract, optional
            Image acquisition object. If None, then use the global instance. Default is None
        update_ms : int, optional
            Update frequency in ms. Default is 20
        highlight_saturation : bool, optional
            To highlight saturation red in image. Default is True

        """
        super().__init__()

        # Get default values
        image_acquisition, _ = scf.get_default_or_global_instances(image_acquisition_default=image_acquisition)

        # Store variables
        self.image_acquisition = image_acquisition
        self.highlight_saturation = highlight_saturation

        # Create figure and axes
        self.fig = plt.figure()
        self._register_plot(self.fig)
        self.ax = self.fig.gca()

        # Create image object
        self.im = self.ax.imshow(self.grab_frame(), cmap="gray")

        # Create animation object (must be defined to variable)
        self.anim = FuncAnimation(self.fig, self.update, interval=update_ms, cache_frame_data=False)

        # Define close function and bind to keystroke
        self.fig.canvas.mpl_connect("key_press_event", self.close)

        # Show figure
        plt.show()

    def close(self, event: KeyEvent) -> None:
        """
        Closes window

        """
        if event.key == "escape":
            print("Closing window")
        # always free the maptlotlib plot
        super().close()

    def update(self, i: int) -> None:
        """
        Frame update function

        """
        self.im.set_data(self.grab_frame())

    def grab_frame(self) -> np.ndarray:
        """
        Frame grabbing function with saturation highlighting

        """
        frame = self.image_acquisition.get_frame()
        if self.highlight_saturation:
            return highlight_saturation(frame, self.image_acquisition.max_value)
        else:
            return frame.astype(np.float32) / np.float32(self.image_acquisition.max_value)
