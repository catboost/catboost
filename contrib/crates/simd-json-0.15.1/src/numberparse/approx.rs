use super::{
    is_integer, is_made_of_eight_digits_fast,
    is_not_structural_or_whitespace_or_exponent_or_decimal, parse_eight_digits_unrolled,
};
use crate::StaticNode;
use crate::charutils::is_structural_or_whitespace;
use crate::macros::{static_cast_i64, unlikely};
use crate::safer_unchecked::GetSaferUnchecked;
use crate::{Deserializer, ErrorType, Result};

const POWER_OF_TEN: [f64; 632] = [
    1e-323, 1e-322, 1e-321, 1e-320, 1e-319, 1e-318, 1e-317, 1e-316, 1e-315, 1e-314, 1e-313, 1e-312,
    1e-311, 1e-310, 1e-309, 1e-308, 1e-307, 1e-306, 1e-305, 1e-304, 1e-303, 1e-302, 1e-301, 1e-300,
    1e-299, 1e-298, 1e-297, 1e-296, 1e-295, 1e-294, 1e-293, 1e-292, 1e-291, 1e-290, 1e-289, 1e-288,
    1e-287, 1e-286, 1e-285, 1e-284, 1e-283, 1e-282, 1e-281, 1e-280, 1e-279, 1e-278, 1e-277, 1e-276,
    1e-275, 1e-274, 1e-273, 1e-272, 1e-271, 1e-270, 1e-269, 1e-268, 1e-267, 1e-266, 1e-265, 1e-264,
    1e-263, 1e-262, 1e-261, 1e-260, 1e-259, 1e-258, 1e-257, 1e-256, 1e-255, 1e-254, 1e-253, 1e-252,
    1e-251, 1e-250, 1e-249, 1e-248, 1e-247, 1e-246, 1e-245, 1e-244, 1e-243, 1e-242, 1e-241, 1e-240,
    1e-239, 1e-238, 1e-237, 1e-236, 1e-235, 1e-234, 1e-233, 1e-232, 1e-231, 1e-230, 1e-229, 1e-228,
    1e-227, 1e-226, 1e-225, 1e-224, 1e-223, 1e-222, 1e-221, 1e-220, 1e-219, 1e-218, 1e-217, 1e-216,
    1e-215, 1e-214, 1e-213, 1e-212, 1e-211, 1e-210, 1e-209, 1e-208, 1e-207, 1e-206, 1e-205, 1e-204,
    1e-203, 1e-202, 1e-201, 1e-200, 1e-199, 1e-198, 1e-197, 1e-196, 1e-195, 1e-194, 1e-193, 1e-192,
    1e-191, 1e-190, 1e-189, 1e-188, 1e-187, 1e-186, 1e-185, 1e-184, 1e-183, 1e-182, 1e-181, 1e-180,
    1e-179, 1e-178, 1e-177, 1e-176, 1e-175, 1e-174, 1e-173, 1e-172, 1e-171, 1e-170, 1e-169, 1e-168,
    1e-167, 1e-166, 1e-165, 1e-164, 1e-163, 1e-162, 1e-161, 1e-160, 1e-159, 1e-158, 1e-157, 1e-156,
    1e-155, 1e-154, 1e-153, 1e-152, 1e-151, 1e-150, 1e-149, 1e-148, 1e-147, 1e-146, 1e-145, 1e-144,
    1e-143, 1e-142, 1e-141, 1e-140, 1e-139, 1e-138, 1e-137, 1e-136, 1e-135, 1e-134, 1e-133, 1e-132,
    1e-131, 1e-130, 1e-129, 1e-128, 1e-127, 1e-126, 1e-125, 1e-124, 1e-123, 1e-122, 1e-121, 1e-120,
    1e-119, 1e-118, 1e-117, 1e-116, 1e-115, 1e-114, 1e-113, 1e-112, 1e-111, 1e-110, 1e-109, 1e-108,
    1e-107, 1e-106, 1e-105, 1e-104, 1e-103, 1e-102, 1e-101, 1e-100, 1e-99, 1e-98, 1e-97, 1e-96,
    1e-95, 1e-94, 1e-93, 1e-92, 1e-91, 1e-90, 1e-89, 1e-88, 1e-87, 1e-86, 1e-85, 1e-84, 1e-83,
    1e-82, 1e-81, 1e-80, 1e-79, 1e-78, 1e-77, 1e-76, 1e-75, 1e-74, 1e-73, 1e-72, 1e-71, 1e-70,
    1e-69, 1e-68, 1e-67, 1e-66, 1e-65, 1e-64, 1e-63, 1e-62, 1e-61, 1e-60, 1e-59, 1e-58, 1e-57,
    1e-56, 1e-55, 1e-54, 1e-53, 1e-52, 1e-51, 1e-50, 1e-49, 1e-48, 1e-47, 1e-46, 1e-45, 1e-44,
    1e-43, 1e-42, 1e-41, 1e-40, 1e-39, 1e-38, 1e-37, 1e-36, 1e-35, 1e-34, 1e-33, 1e-32, 1e-31,
    1e-30, 1e-29, 1e-28, 1e-27, 1e-26, 1e-25, 1e-24, 1e-23, 1e-22, 1e-21, 1e-20, 1e-19, 1e-18,
    1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
    1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13,
    1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29,
    1e30, 1e31, 1e32, 1e33, 1e34, 1e35, 1e36, 1e37, 1e38, 1e39, 1e40, 1e41, 1e42, 1e43, 1e44, 1e45,
    1e46, 1e47, 1e48, 1e49, 1e50, 1e51, 1e52, 1e53, 1e54, 1e55, 1e56, 1e57, 1e58, 1e59, 1e60, 1e61,
    1e62, 1e63, 1e64, 1e65, 1e66, 1e67, 1e68, 1e69, 1e70, 1e71, 1e72, 1e73, 1e74, 1e75, 1e76, 1e77,
    1e78, 1e79, 1e80, 1e81, 1e82, 1e83, 1e84, 1e85, 1e86, 1e87, 1e88, 1e89, 1e90, 1e91, 1e92, 1e93,
    1e94, 1e95, 1e96, 1e97, 1e98, 1e99, 1e100, 1e101, 1e102, 1e103, 1e104, 1e105, 1e106, 1e107,
    1e108, 1e109, 1e110, 1e111, 1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119, 1e120,
    1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127, 1e128, 1e129, 1e130, 1e131, 1e132, 1e133,
    1e134, 1e135, 1e136, 1e137, 1e138, 1e139, 1e140, 1e141, 1e142, 1e143, 1e144, 1e145, 1e146,
    1e147, 1e148, 1e149, 1e150, 1e151, 1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159,
    1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167, 1e168, 1e169, 1e170, 1e171, 1e172,
    1e173, 1e174, 1e175, 1e176, 1e177, 1e178, 1e179, 1e180, 1e181, 1e182, 1e183, 1e184, 1e185,
    1e186, 1e187, 1e188, 1e189, 1e190, 1e191, 1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198,
    1e199, 1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207, 1e208, 1e209, 1e210, 1e211,
    1e212, 1e213, 1e214, 1e215, 1e216, 1e217, 1e218, 1e219, 1e220, 1e221, 1e222, 1e223, 1e224,
    1e225, 1e226, 1e227, 1e228, 1e229, 1e230, 1e231, 1e232, 1e233, 1e234, 1e235, 1e236, 1e237,
    1e238, 1e239, 1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247, 1e248, 1e249, 1e250,
    1e251, 1e252, 1e253, 1e254, 1e255, 1e256, 1e257, 1e258, 1e259, 1e260, 1e261, 1e262, 1e263,
    1e264, 1e265, 1e266, 1e267, 1e268, 1e269, 1e270, 1e271, 1e272, 1e273, 1e274, 1e275, 1e276,
    1e277, 1e278, 1e279, 1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287, 1e288, 1e289,
    1e290, 1e291, 1e292, 1e293, 1e294, 1e295, 1e296, 1e297, 1e298, 1e299, 1e300, 1e301, 1e302,
    1e303, 1e304, 1e305, 1e306, 1e307, 1e308,
];

impl Deserializer<'_> {
    /// called by `parse_number` when we know that the output is a float,
    /// but where there might be some integer overflow. The trick here is to
    /// parse using floats from the start.
    /// Do not call this function directly as it skips some of the checks from
    /// `parse_number`
    ///
    /// This function will almost never be called!!!
    ///
    /// Note: a redesign could avoid this function entirely.
    ///
    #[inline(never)]
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::too_many_lines
    )]
    fn parse_float(idx: usize, p: &[u8], negative: bool) -> Result<StaticNode> {
        let mut digitcount = usize::from(negative);
        let mut i: f64;
        let mut digit: u8;
        let mut d;
        if unsafe { *p.get_kinda_unchecked(digitcount) } == b'0' {
            // 0 cannot be followed by an integer
            digitcount += 1;
            i = 0.0;
        } else {
            digit = unsafe { *p.get_kinda_unchecked(digitcount) } - b'0';
            i = f64::from(digit);
            digitcount += 1;
            while is_integer(unsafe { *p.get_kinda_unchecked(digitcount) }) {
                digit = unsafe { *p.get_kinda_unchecked(digitcount) } - b'0';
                i = 10.0 * i + f64::from(digit);
                digitcount += 1;
            }
        }
        if unsafe { *p.get_kinda_unchecked(digitcount) } == b'.' {
            let mut fraction: u64 = 0;
            let mut fraction_weight: u64 = 10;
            digitcount += 1;
            //let mut fractionalweight: f64 = 1.0;
            d = unsafe { *p.get_kinda_unchecked(digitcount) };
            if is_integer(d) {
                digit = d - b'0';
                digitcount += 1;
                fraction += u64::from(digit);
            } else {
                return Err(Self::error_c(
                    idx + digitcount,
                    d as char,
                    ErrorType::InvalidNumber,
                ));
            }

            while is_integer(unsafe { *p.get_kinda_unchecked(digitcount) })
                && fraction_weight <= 10_000_000_000_000_000_u64
            {
                digit = unsafe { *p.get_kinda_unchecked(digitcount) } - b'0';
                digitcount += 1;
                fraction_weight *= 10;
                fraction = fraction * 10 + u64::from(digit);
            }

            let mut fraction_weight = fraction_weight as f64;
            let mut fraction = (fraction as f64) / fraction_weight;

            while is_integer(unsafe { *p.get_kinda_unchecked(digitcount) }) {
                digit = unsafe { *p.get_kinda_unchecked(digitcount) } - b'0';
                digitcount += 1;
                fraction_weight *= 10.0;
                fraction += f64::from(digit) / fraction_weight;
            }
            i += fraction;
        }
        if (unsafe { *p.get_kinda_unchecked(digitcount) } == b'e')
            || (unsafe { *p.get_kinda_unchecked(digitcount) } == b'E')
        {
            digitcount += 1;
            let mut negexp: bool = false;
            if unsafe { *p.get_kinda_unchecked(digitcount) } == b'-' {
                negexp = true;
                digitcount += 1;
            } else if unsafe { *p.get_kinda_unchecked(digitcount) } == b'+' {
                digitcount += 1;
            }
            d = unsafe { *p.get_kinda_unchecked(digitcount) };
            if !is_integer(d) {
                return Err(Self::error_c(
                    idx + digitcount,
                    d as char,
                    ErrorType::InvalidNumber,
                ));
            }
            digit = unsafe { *p.get_kinda_unchecked(digitcount) } - b'0';
            let mut expnumber: u32 = u32::from(digit); // exponential part
            digitcount += 1;
            d = unsafe { *p.get_kinda_unchecked(digitcount) };
            if is_integer(d) {
                digit = d - b'0';
                expnumber = 10 * expnumber + u32::from(digit);
                digitcount += 1;
            }
            d = unsafe { *p.get_kinda_unchecked(digitcount) };
            if is_integer(d) {
                digit = d - b'0';
                expnumber = 10 * expnumber + u32::from(digit);
                digitcount += 1;
            }
            d = unsafe { *p.get_kinda_unchecked(digitcount) };
            if is_integer(d) {
                digit = d - b'0';
                expnumber = 10 * expnumber + u32::from(digit);
                digitcount += 1;
            }
            d = unsafe { *p.get_kinda_unchecked(digitcount) };
            if is_integer(d) {
                // we refuse to parse this
                return Err(Self::error_c(
                    idx + digitcount,
                    d as char,
                    ErrorType::InvalidNumber,
                ));
            }
            let exponent: i32 = if negexp {
                -(expnumber as i32)
            } else {
                expnumber as i32
            };
            if !(-323..=308).contains(&exponent) {
                // we refuse to parse this
                return Err(Self::error_c(
                    idx + digitcount,
                    d as char,
                    ErrorType::InvalidExponent,
                ));
            }
            i *= unsafe { POWER_OF_TEN.get_kinda_unchecked((323 + exponent) as usize) };
        }

        d = unsafe { *p.get_kinda_unchecked(digitcount) };
        if is_structural_or_whitespace(d) == 0 {
            Err(Self::error_c(
                idx + digitcount,
                d as char,
                ErrorType::InvalidNumber,
            ))
        } else {
            Ok(StaticNode::from(if negative { -i } else { i }))
        }
    }

    /// called by `parse_number` when we know that the output is an integer,
    /// but where there might be some integer overflow.
    /// we want to catch overflows!
    /// Do not call this function directly as it skips some of the checks from
    /// `parse_number`
    ///
    /// This function will almost never be called!!!
    ///
    #[cfg(not(feature = "128bit"))]
    #[inline(never)]
    #[allow(clippy::cast_possible_wrap)]
    fn parse_large_integer(idx: usize, buf: &[u8], negative: bool) -> Result<StaticNode> {
        let mut digitcount = usize::from(negative);
        let mut i: u64;
        let mut d = unsafe { *buf.get_kinda_unchecked(digitcount) };
        let mut digit: u8;

        if d == b'0' {
            digitcount += 1;
            d = unsafe { *buf.get_kinda_unchecked(digitcount) };
            i = 0;
        } else {
            digit = d - b'0';
            i = u64::from(digit);
            digitcount += 1;
            d = unsafe { *buf.get_kinda_unchecked(digitcount) };
            // the is_made_of_eight_digits_fast routine is unlikely to help here because
            // we rarely see large integer parts like 123456789
            while is_integer(d) {
                digit = d - b'0';
                if let Some(i1) = i
                    .checked_mul(10)
                    .and_then(|i| i.checked_add(u64::from(digit)))
                {
                    i = i1;
                } else {
                    return Err(Self::error_c(
                        idx + digitcount,
                        d as char,
                        ErrorType::Overflow,
                    ));
                }
                digitcount += 1;
                d = unsafe { *buf.get_kinda_unchecked(digitcount) };
            }
        }

        if negative && i > 9_223_372_036_854_775_808 {
            //i64::MIN * -1
            return Err(Self::error_c(
                idx + digitcount,
                d as char,
                ErrorType::Overflow,
            ));
        }

        if is_structural_or_whitespace(d) == 0 {
            Err(Self::error_c(
                idx + digitcount,
                d as char,
                ErrorType::InvalidNumber,
            ))
        } else if negative {
            unsafe { Ok(StaticNode::I64(static_cast_i64!(i.wrapping_neg()))) }
        } else {
            Ok(StaticNode::U64(i))
        }
    }

    #[cfg(feature = "128bit")]
    #[inline(never)]
    #[allow(clippy::cast_possible_wrap)]
    fn parse_large_integer(idx: usize, buf: &[u8], negative: bool) -> Result<StaticNode> {
        let mut digitcount = usize::from(negative);
        let mut i: u128;
        let mut d = unsafe { *buf.get_kinda_unchecked(digitcount) };
        let mut digit: u8;

        if d == b'0' {
            digitcount += 1;
            d = unsafe { *buf.get_kinda_unchecked(digitcount) };
            i = 0;
        } else {
            digit = d - b'0';
            i = u128::from(digit);
            digitcount += 1;
            d = unsafe { *buf.get_kinda_unchecked(digitcount) };
            // the is_made_of_eight_digits_fast routine is unlikely to help here because
            // we rarely see large integer parts like 123456789
            while is_integer(d) {
                digit = d - b'0';
                if let Some(i1) = i
                    .checked_mul(10)
                    .and_then(|i| i.checked_add(u128::from(digit)))
                {
                    i = i1;
                } else {
                    return Err(Self::error_c(
                        idx + digitcount,
                        d as char,
                        ErrorType::Overflow,
                    ));
                }
                digitcount += 1;
                d = unsafe { *buf.get_kinda_unchecked(digitcount) };
            }
        }

        if negative && i > 170_141_183_460_469_231_731_687_303_715_884_105_728_u128 {
            //i64::MIN * -1
            return Err(Self::error_c(
                idx + digitcount,
                d as char,
                ErrorType::Overflow,
            ));
        }

        if is_structural_or_whitespace(d) == 0 {
            Err(Self::error_c(
                idx + digitcount,
                d as char,
                ErrorType::InvalidNumber,
            ))
        } else if negative {
            let i = unsafe { static_cast_i128!(i.wrapping_neg()) };
            if let Ok(i) = i64::try_from(i) {
                Ok(StaticNode::I64(i))
            } else {
                Ok(StaticNode::I128(i))
            }
        } else if let Ok(i) = u64::try_from(i) {
            Ok(StaticNode::U64(i))
        } else {
            Ok(StaticNode::U128(i))
        }
    }

    // parse the number at buf + offset
    // define JSON_TEST_NUMBERS for unit testing
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    pub(crate) fn parse_number(idx: usize, buf: &[u8], negative: bool) -> Result<StaticNode> {
        let buf = unsafe { buf.get_kinda_unchecked(idx..) };
        let mut byte_count = usize::from(negative);
        let mut ignore_count: u8 = 0;
        //let startdigits: *const u8 = p;
        let mut i: u64;
        let mut d = unsafe { *buf.get_kinda_unchecked(byte_count) };
        let mut digit: u8;
        if d == b'0' {
            // 0 cannot be followed by an integer
            byte_count += 1;
            d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            if is_not_structural_or_whitespace_or_exponent_or_decimal(d) {
                return Err(Self::error_c(
                    idx + byte_count,
                    d as char,
                    ErrorType::InvalidNumber,
                ));
            }
            i = 0;
        } else {
            if !is_integer(d) {
                // must start with an integer
                return Err(Self::error_c(
                    idx + byte_count,
                    d as char,
                    ErrorType::InvalidNumber,
                ));
            }
            digit = d - b'0';
            i = u64::from(digit);
            byte_count += 1;

            d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            // the is_made_of_eight_digits_fast routine is unlikely to help here because
            // we rarely see large integer parts like 123456789
            while is_integer(d) {
                digit = d - b'0';
                i = i.wrapping_mul(10);
                if let Some(i1) = i.checked_add(u64::from(digit)) {
                    i = i1;
                } else {
                    return Err(Self::error_c(
                        idx + byte_count,
                        d as char,
                        ErrorType::Overflow,
                    ));
                }
                //i = 10 * i + u64::from(digit); // might overflow
                byte_count += 1;
                d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            }
        }

        let mut exponent: i64 = if d == b'.' {
            ignore_count += 1;
            byte_count += 1;
            d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            let firstafterperiod = byte_count;
            if is_integer(d) {
                digit = d - b'0';
                byte_count += 1;
                i = i.wrapping_mul(10).wrapping_add(u64::from(digit));
            } else {
                return Err(Self::error_c(
                    idx + byte_count,
                    d as char,
                    ErrorType::InvalidNumber,
                ));
            }
            // this helps if we have lots of decimals!
            // this turns out to be frequent enough.

            #[cfg(feature = "swar-number-parsing")]
            {
                let chars: [u8; 8] = unsafe {
                    *(buf
                        .get_kinda_unchecked(byte_count..byte_count + 8)
                        .as_ptr()
                        .cast())
                };

                if is_made_of_eight_digits_fast(chars) {
                    i = i.wrapping_mul(100_000_000).wrapping_add(u64::from(
                        parse_eight_digits_unrolled(unsafe {
                            buf.get_kinda_unchecked(byte_count..)
                        }),
                    ));
                    byte_count += 8;
                }
            }
            d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            while is_integer(d) {
                digit = d - b'0';
                i = i.wrapping_mul(10).wrapping_add(u64::from(digit));
                byte_count += 1;
                d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            }
            firstafterperiod as i64 - byte_count as i64
        } else {
            0
        };
        let mut expnumber: i16 = 0; // exponential part
        if (d == b'e') || (d == b'E') {
            ignore_count += 1;
            byte_count += 1;
            d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            let mut negexp: bool = false;
            if d == b'-' {
                negexp = true;
                ignore_count += 1;
                byte_count += 1;
                d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            } else if d == b'+' {
                ignore_count += 1;
                byte_count += 1;
                d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            }
            if !is_integer(d) {
                return Err(Self::error_c(
                    idx + byte_count,
                    d as char,
                    ErrorType::InvalidNumber,
                ));
            }
            digit = d - b'0';
            expnumber = i16::from(digit);
            byte_count += 1;
            ignore_count += 1;
            d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            if is_integer(d) {
                digit = d - b'0';
                expnumber = 10 * expnumber + i16::from(digit);
                ignore_count += 1;
                byte_count += 1;
                d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            }
            if is_integer(d) {
                digit = d - b'0';
                expnumber = 10 * expnumber + i16::from(digit);
                ignore_count += 1;
                byte_count += 1;
                d = unsafe { *buf.get_kinda_unchecked(byte_count) };
            }
            if is_integer(d) {
                // we refuse to parse this
                return Err(Self::error_c(
                    idx + byte_count,
                    d as char,
                    ErrorType::InvalidNumber,
                ));
            }
            exponent += i64::from(if negexp { -expnumber } else { expnumber });
        }
        let v = if (exponent != 0) || (expnumber != 0) {
            if unlikely!((byte_count - ignore_count as usize) >= 19) {
                // this is uncommon!!!
                // this is almost never going to get called!!!
                // we start anew, going slowly!!!
                return Self::parse_float(idx, buf, negative);
            }
            ///////////
            // We want 0.1e1 to be a float.
            //////////
            if i == 0 {
                StaticNode::from(0.0)
            } else {
                if !(-323..=308).contains(&exponent) {
                    return Self::parse_float(idx, buf, negative);
                }

                let mut d1: f64 = i as f64;
                d1 *= unsafe { POWER_OF_TEN.get_kinda_unchecked((323 + exponent) as usize) };
                StaticNode::from(if negative { d1 * -1.0 } else { d1 })
            }
        } else {
            if unlikely!(byte_count >= 18) {
                // this is uncommon!!!
                return Self::parse_large_integer(idx, buf, negative);
            }
            if negative {
                unsafe { StaticNode::I64(static_cast_i64!(i.wrapping_neg())) }
            } else {
                StaticNode::U64(i)
            }
        };
        if is_structural_or_whitespace(d) == 0 {
            Err(Self::error_c(
                idx + byte_count,
                d as char,
                ErrorType::InvalidNumber,
            ))
        } else {
            Ok(v)
        }
    }
}

#[cfg(test)]
mod test {
    #![allow(clippy::default_trait_access)]
    use crate::value::owned::to_value;
    use float_cmp::approx_eq;
    use value_trait::prelude::*;

    #[test]
    fn bad_exp() {
        let mut too_big = String::from("1e309");
        let mut too_big = unsafe { too_big.as_bytes_mut() };
        let v_too_big = to_value(&mut too_big);
        assert!(v_too_big.is_err());
        let mut too_small = String::from("1e-324");
        let mut too_small = unsafe { too_small.as_bytes_mut() };
        let v_too_small = to_value(&mut too_small);
        assert!(v_too_small.is_err());
        let mut too_big = String::from("1e1000");
        let mut too_big = unsafe { too_big.as_bytes_mut() };
        let v_too_big = to_value(&mut too_big);
        assert!(v_too_big.is_err());
        let mut too_small = String::from("1e-1000");
        let mut too_small = unsafe { too_small.as_bytes_mut() };
        let v_too_small = to_value(&mut too_small);
        assert!(v_too_small.is_err());

        let mut too_big = String::from("100000000000000000000000000000000000000000000e309");
        let mut too_big = unsafe { too_big.as_bytes_mut() };
        let v_too_big = to_value(&mut too_big);
        assert!(v_too_big.is_err());
        let mut too_small = String::from("100000000000000000000000000000000000000000000e-324");
        let mut too_small = unsafe { too_small.as_bytes_mut() };
        let v_too_small = to_value(&mut too_small);
        assert!(v_too_small.is_err());
        let mut too_big = String::from("100000000000000000000000000000000000000000000e1000");
        let mut too_big = unsafe { too_big.as_bytes_mut() };
        let v_too_big = to_value(&mut too_big);
        assert!(v_too_big.is_err());
        let mut too_small = String::from("100000000000000000000000000000000000000000000e-1000");
        let mut too_small = unsafe { too_small.as_bytes_mut() };
        let v_too_small = to_value(&mut too_small);
        assert!(v_too_small.is_err());
    }

    #[test]
    fn bad_dot() {
        let mut i = String::from("1.");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i);
        assert!(r.is_err());
        let mut i = String::from("1.e");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i);
        assert!(r.is_err());
        let mut i = String::from("100000000000000000000000000000000000000000000.");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i);
        assert!(r.is_err());
        let mut i = String::from("100000000000000000000000000000000000000000000.e");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i);
        assert!(r.is_err());
    }

    #[test]
    fn bad_e() {
        let mut i = String::from("1.0e");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i);
        assert!(r.is_err());
        let mut i = String::from("1.0e");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i);
        assert!(r.is_err());
        let mut i = String::from("100000000000000000000000000000000000000000000.0e");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i);
        assert!(r.is_err());
        let mut i = String::from("100000000000000000000000000000000000000000000.0ee");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i);
        assert!(r.is_err());
    }

    #[test]
    fn zero() {
        let mut i = String::from("0");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i);
        assert_eq!(r.expect("0"), 0);
    }

    #[test]
    fn float_zero() {
        let mut i = String::from("0e1");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i).expect("failed to decode");
        assert!(approx_eq!(f64, r.as_f64().expect("float"), 0.0));
    }

    #[test]
    fn minus_309() {
        let mut i = String::from("-5.96916642387374e-309");
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i).expect("failed to decode");
        assert!(approx_eq!(
            f64,
            r.as_f64().expect("float"),
            -5.969_166_423_873_74e-309
        ));
    }
    #[allow(clippy::unreadable_literal)]
    #[test]
    fn tiny_float() {
        let mut i = String::from(
            "-0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000596916642387374",
        );
        let mut i = unsafe { i.as_bytes_mut() };
        let r = to_value(&mut i).expect("failed to decode");
        assert!(approx_eq!(f64, r.as_f64().expect("float"), -0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000596916642387374));
    }
}
