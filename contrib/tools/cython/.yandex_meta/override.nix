pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "3.2.5";

  src = fetchPypi {
    pname = "cython";
    inherit version;
    hash = "sha256-PdQuTPNq0V8mW9/sIzfMAMaIyOttN0/9E7sZQ3wnu6E=";
  };

  patches = [];
}
