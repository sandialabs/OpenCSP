#!/bin/bash
echo "Usage example: "
echo "    ./gen_library_ref_rst_from_class_list.sh \\"
echo "    ./source/library_reference/common/lib/csp \\"
echo "    CSP \\"
echo "    \"This is a collection of CSP utilities for OpenCSP." "opencsp.common.lib.csp.Facet, opencsp.common.lib.csp.FacetEnsemble, opencsp.common.lib.csp.HeliostatAbstract, opencsp.common.lib.csp.HeliostatAzEl, opencsp.common.lib.csp.HeliostatConfiguration, opencsp.common.lib.csp.LightPath, opencsp.common.lib.csp.LightPathEnsemble, opencsp.common.lib.csp.LightSource, opencsp.common.lib.csp.LightSourcePoint, opencsp.common.lib.csp.LightSourceSun, opencsp.common.lib.csp.MirrorAbstract, opencsp.common.lib.csp.MirrorParametric, opencsp.common.lib.csp.MirrorParametricRectangular, opencsp.common.lib.csp.MirrorPoint, opencsp.common.lib.csp.OpticOrientationAbstract, opencsp.common.lib.csp.RayTrace, opencsp.common.lib.csp.RayTraceable, opencsp.common.lib.csp.Scene, opencsp.common.lib.csp.SolarField, opencsp.common.lib.csp.StandardPlotOutput, opencsp.common.lib.csp.Tower, opencsp.common.lib.csp.VisualizeOrthorectifiedSlopeAbstract, opencsp.common.lib.csp.sun_position, opencsp.common.lib.csp.sun_track, opencsp.common.lib.csp.visualize_orthorectified_image\"\\"

DOC_PATH="${1:?Error: DOC_PATH is unset}"
DOC_SHORT_DESCRIPTOR="${2:?Error: DOC_SHORT_DESCRIPTOR is unset}"
DOC_LONG_DESCRIPTOR="${3:?Error: DOC_LONG_DESCRIPTOR is unset}"
CLASS_LIST="${4:?Error: CLASS_LIST is unset}"

sed_cmd=sed
which gsed &> /dev/null
ret=$?
if [ $ret -eq 0 ]; then
  sed_cmd=gsed
fi

mkdir -p $DOC_PATH || true
rm -i $DOC_PATH/config.rst
rm -i $DOC_PATH/index.rst

for class in $(echo $CLASS_LIST); do
  sans_comma=$(echo $class | tr -d ',')
  cat template_config | ${sed_cmd} "s/__MODULE__/$sans_comma/g" >> $DOC_PATH/config.rst
done

echo "$DOC_SHORT_DESCRIPTOR" >> $DOC_PATH/index.rst
footer_len=$(echo $DOC_SHORT_DESCRIPTOR | wc -c)
python -c "print('=' * ${footer_len})" >> $DOC_PATH/index.rst
echo "" >> $DOC_PATH/index.rst

echo "$DOC_LONG_DESCRIPTOR" >> $DOC_PATH/index.rst
echo "" >> $DOC_PATH/index.rst
cat template_index >> $DOC_PATH/index.rst

echo -n "$DOC_PATH" | awk -F 'library_reference/' '{print "   "$2"/index.rst"}' >> ./source/library_reference/index.rst
