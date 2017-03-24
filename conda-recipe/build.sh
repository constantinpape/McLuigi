# cf. https://github.com/ilastik/nature_methods_multicut_pipeline/blob/master/build/multicut_pipeline/build.sh
mkdir -p ${PREFIX}/mcLuigi

# copy all the python sources
cp  mcLuigi/*Tasks.py ${PREFIX}/mcLuigi
cp mcLuigi/__init__.py ${PREFIX}/mcLuigi
cp -r mcLuigi/tools ${PREFIX}/mcLuigi

echo "${PREFIX}/mcLuigi" > ${PREFIX}/lib/python2.7/site-packages/mcLuigi.pth
python -m compileall ${PREFIX}/mcLuigi
