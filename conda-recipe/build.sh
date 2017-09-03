# cf. https://github.com/ilastik/nature_methods_multicut_pipeline/blob/master/build/multicut_pipeline/build.sh
mkdir -p ${PREFIX}/mc_luigi

# copy all the python sources
cp mc_luigi/*.py ${PREFIX}/mc_luigi
cp -r mc_luigi/tools ${PREFIX}/mc_luigi

echo "${PREFIX}" > ${PREFIX}/lib/python${PY_VER}/site-packages/mc_luigi.pth
python -m compileall ${PREFIX}/mc_luigi
