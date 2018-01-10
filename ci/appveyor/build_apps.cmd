call ya.bat make -o . --verbose --stat -r catboost\app -DCUDA_ROOT="C:\\CUDA\\v8.0"
mv catboost\app\catboost.exe catboost\app\catboost-cuda.exe
python ci\webdav_upload.py catboost\app\catboost-cuda.exe

call ya.bat make -o . --verbose --stat -r catboost\app
python ci\webdav_upload.py catboost\app\catboost.exe

