pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "3.2.8";

  src = fetchPypi {
    pname = "cython";
    inherit version;
    hash = "sha256-9PI6VrJSIaBvkYF/6PMRSri0ik+scxh9u2S8LEqHlh8=";
  };

  patches = [];
}
