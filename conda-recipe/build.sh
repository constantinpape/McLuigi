# cf. https://github.com/ilastik/nature_methods_multicut_pipeline/blob/master/build/multicut_pipeline/build.sh
mkdir -p ${PREFIX}/mc_luigi

# copy all the python sources
cp mc_luigi/*Tasks.py ${PREFIX}/mc_luigi
cp mc_luigi/__init__.py ${PREFIX}/mc_luigi
cp -r mc_luigi/tools ${PREFIX}/mc_luigi

echo "${PREFIX}/mc_luigi" > ${PREFIX}/lib/python2.7/site-packages/mc_luigi.pth
python -m compileall ${PREFIX}/mc_luigi
