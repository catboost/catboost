script_folder="/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/conan/build/Release/generators"
echo "echo Restoring environment" > "$script_folder/deactivate_conanbuildenv-release-armv8.sh"
for v in PATH SWIG_LIB LD_LIBRARY_PATH DYLD_LIBRARY_PATH
do
    is_defined="true"
    value=$(printenv $v) || is_defined="" || true
    if [ -n "$value" ] || [ -n "$is_defined" ]
    then
        echo export "$v='$value'" >> "$script_folder/deactivate_conanbuildenv-release-armv8.sh"
    else
        echo unset $v >> "$script_folder/deactivate_conanbuildenv-release-armv8.sh"
    fi
done


export PATH="/Users/makar/.conan2/p/yasmf5f5a1edf2bac/p/bin:/Users/makar/.conan2/p/swig730af938a1133/p/bin:/Users/makar/.conan2/p/ragel4086a4d35af21/p/bin:$PATH"
export SWIG_LIB="/Users/makar/.conan2/p/swig730af938a1133/p/bin/swiglib"
export LD_LIBRARY_PATH="/Users/makar/.conan2/p/swig730af938a1133/p/lib:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/Users/makar/.conan2/p/swig730af938a1133/p/lib:$DYLD_LIBRARY_PATH"