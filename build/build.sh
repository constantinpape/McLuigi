# cf. https://github.com/ilastik/nature_methods_multicut_pipeline/blob/master/build/multicut_pipeline/build.sh
# copy all the python sources
mkdir -p ${PREFIX}/mc_luigi
cp -r mcLuigi/* ${PREFIX}/mc_luigi
# TODO is this compiling necessary?
python -m compileall ${PREFIX}/mc_luigi
