import opencsp.common.lib.tool.log_tools as lt


class RenderControlPowerpointSlide:
    """Controls for how to render a power slide."""

    def __init__(
        self,
        title_size=30,
        title_location: tuple[float, float] = (0.48, 0.25),
        text_size=16,
        text_location: tuple[float, float] = (0.53, 0.83),
        image_caption_size=14,
        is_title_slide=False,
        slide_index=-1,
        slide_size: tuple[float, float] = (13.33, 7.5),
        inter_cell_buffer: float = 0.35,
        reduced_image_size_scale: float = -1,
    ):
        """Controls for how to render a power slide.

        Args:
            title_size (int): Font size of the title text box. Defaults to 30.
            title_location (tuple): Top-left location of the title text box in inches. Defaults to (.48,.25).
            text_size (int): Font size of the content text box. Defaults to 16.
            text_location (tuple): Top-left location of the content text box in inches. Defaults to (.53,.83).
            image_caption_size (int): Font size of image captions. Defaults to 14.
            is_title_slide (bool): True for the title slide, False for all content slides. Defaults to False.
            slide_index (int): Where to put this slide in the presentation, or -1 to put it at the end. Defaults to -1.
            slide_size (tuple): Size of the slide in inches. Defaults to (13.33,7.5).
            inter_cell_buffer (float): Default amount of space between cells in inches, for slides that are arranged in a grid.
            reduced_image_size_scale (float): How much to reduce image size to relative to its rendered size, assuming 300 dpi,
                                              in order to save on space. -1 for no reduction. Defaults to -1.
        """
        self.title_size = title_size
        """ Size of the title font, in pnts """
        self.title_location = title_location
        """ Left, top location of title, in inches """
        self.text_size = text_size
        """ Size of the text font, in pnts """
        self.text_location = text_location
        """ Default left, top location of text, in inches """
        self.image_caption_size = image_caption_size
        """ Size of the image caption font, in pnts """
        self.is_title_slide = is_title_slide
        """ True if this is the title slide, or False otherwise """
        self.slide_index = slide_index
        """ Index to insert into the powerpoint, or -1 to insert in add order. """
        self.slide_size = slide_size
        """ Width, height of the slide, in inches """
        self.inter_cell_buffer = inter_cell_buffer
        """ Space between cells, when generating slides with PowerpointSlide.template_content_grid() """
        self.reduced_image_size_scale = reduced_image_size_scale
        """ When images are reduced to save on disk space, this is the cutoff threshold at which images are scaled down. """
        self.slide_dpi = 300
        """ Assumed dots per inch of the slide """

    @classmethod
    def title_slide(cls):
        """Default settings for the first frame slide"""
        return cls(24, (0.77, 5.7), 18, (0.77, 6.3), is_title_slide=True)

    def get_title_dims(self):
        """Get the [x,y,width,height] of the title text box."""
        return [*self.title_location, self.slide_size[0] - self.title_location[0], 0.62]

    def get_text_dims(self):
        """Get the [x,y,width,height] of the contents text box."""
        return [
            *self.text_location,
            self.slide_size[0] - self.text_location[0],
            self.slide_size[1] - self.text_location[1],
        ]
