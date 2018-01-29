call ya.bat make -r -T -o . .\catboost\R-package\src\
cd catboost\R-package
mkdir catboost
cp DESCRIPTION catboost
cp NAMESPACE catboost
cp README.md catboost

cp -r R catboost
cp -r inst catboost
cp -r man catboost
cp -r tests catboost

mkdir catboost\inst\libs
mkdir catboost\inst\libs\x64
cp src\libcatboostr.dll catboost\inst\libs\x64

7z -ttar a dummy catboost -so | 7z -si -tgzip a catboost-R-Windows.tgz

cd ..\..
dir catboost\R-package
python ci\webdav_upload.py catboost\R-package\catboost-R-Windows.tgz
