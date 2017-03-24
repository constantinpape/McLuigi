# cf. https://github.com/ilastik/nature_methods_multicut_pipeline/blob/master/build/multicut_pipeline/build.sh
# copy all the python sources
mkdir -p ${PREFIX}/mcLuigi
cp -r mcLuigi/* ${PREFIX}/mcLuigi
echo "${PREFIX}/mcLuigi" > ${PREFIX}/lib/python2.7/site-packages/mcLuigi.pth
python -m compileall ${PREFIX}/mcLuigi
