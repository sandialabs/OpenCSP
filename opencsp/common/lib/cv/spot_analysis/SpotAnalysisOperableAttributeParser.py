import opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor as asaip
import opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable as sao
import opencsp.common.lib.render.ImageAttributeParser as iap
import opencsp.common.lib.tool.typing_tools as tt


class SpotAnalysisOperableAttributeParser(iap.ImageAttributeParser):
    def __init__(self, operable: sao.SpotAnalysisOperable = None, spot_analysis=None):
        # get the current image source path, and initialize the parent
        current_image_source = tt.default(lambda: operable.primary_image.source_path, None)
        super().__init__(current_image_source=current_image_source)

        # set values based on inputs + retrieved attributes
        image_processors: list[asaip.AbstractSpotAnalysisImagesProcessor] = tt.default(lambda: spot_analysis.image_processors, [])
        self.spot_analysis: str = tt.default(lambda: spot_analysis.name, None)
        self.image_processors: list[str] = [processor.name for processor in image_processors]

        # retrieve any available attributes from the associated attributes file
        if self._previous_attr != None:
            self.set_defaults(self._previous_attr.get_parser(SpotAnalysisOperableAttributeParser))

    def attributes_key(self) -> str:
        return "spot analysis attributes"

    def set_defaults(self, other: 'SpotAnalysisOperableAttributeParser'):
        self.spot_analysis = tt.default(self.spot_analysis, other.spot_analysis)
        self.image_processors = tt.default(self.image_processors, other.image_processors)
        super().set_defaults(other)

    def has_contents(self) -> bool:
        if (self.spot_analysis is not None) or \
           (len(self.image_processors) > 0):
            return True
        return super().has_contents()

    def parse_my_contents(self, file_path_name_ext: str, raw_contents: str, my_contents: any):
        self.spot_analysis = my_contents['spot_analysis_name']
        self.image_processors = my_contents['image_processors']
        super().parse_my_contents(file_path_name_ext, raw_contents, my_contents)

    def my_contents_to_json(self, file_path_name_ext: str) -> any:
        ret = {
            'spot_analysis_name': self.spot_analysis,
            'image_processors': self.image_processors
        }
        ret = {**ret, **super().my_contents_to_json(file_path_name_ext)}
        return ret


SpotAnalysisOperableAttributeParser.RegisterClass()
