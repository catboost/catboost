use super::reverse_grapheme_clusters_in_place;
use quickcheck::quickcheck;
use unicode_segmentation::UnicodeSegmentation;

extern crate std;
use self::std::string::String;
use self::std::string::ToString;

fn test_rev(a: &str, b: &str) {
    let mut a = a.to_string();
    reverse_grapheme_clusters_in_place(&mut a);
    assert_eq!(a, b);
}

#[test]
fn test_empty() {
    test_rev("", "");
}

#[test]
fn test_ascii() {
    test_rev("Hello", "olleH");
}

#[test]
fn test_utf8() {
    test_rev("¡Hola!", "!aloH¡");
}

#[test]
fn test_emoji() {
    test_rev("\u{1F36D}\u{1F36E}", "\u{1F36E}\u{1F36D}");
}

#[test]
fn test_combining_mark() {
    test_rev("man\u{0303}ana", "anan\u{0303}am");
}

quickcheck! {
    fn quickchecks(s: String) -> bool {
        let mut in_place = s.clone();
        reverse_grapheme_clusters_in_place(&mut in_place);
        let normal = s.graphemes(true).rev().collect::<String>();
        in_place == normal
    }
}
