{ pkgs ? import <nixpkgs> {} }:
let 
  async-thrift = pkgs.stdenv.mkDerivation {
    pname = "thrift";
    version = "0.15.0";
    src = builtins.fetchGit {
      url = "https://github.com/coastalwhite/thrift";
      ref = "safe";
      rev = "7bd2f1a3237bfe20298b3dec14ca508da378edca";
    };

    buildInputs = with pkgs; [
      flex
      bison
    ];

    configurePhase = ''
      cd compiler/cpp
      mkdir build
      cd build
      ${pkgs.cmake}/bin/cmake ..
      cd ../../..
    '';

    buildPhase = ''
      make -C compiler/cpp/build
    '';

    installPhase = ''
      mkdir -p $out/bin
      mv compiler/cpp/build/bin/thrift $out/bin
    '';
  };
  parquet_thrift_definitions = builtins.fetchGit {
    url = "https://github.com/apache/parquet-format";
    ref = "master";
    rev = "4f208158dba80ff4bff4afaa4441d7270103dff6";
  };
  generate_parquet_format = pkgs.writeShellScriptBin "generate_parquet_format" ''
    thrift --gen rs ${parquet_thrift_definitions}/src/main/thrift/parquet.thrift
  '';
in pkgs.mkShell {
  packages = [ async-thrift generate_parquet_format ];
}