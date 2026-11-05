README for libtiff developpers
==============================

## About TIFFGetMaxCompressionRatio()

That function was introduced in libtiff 4.7.2 per https://gitlab.com/libtiff/libtiff/-/merge_requests/872

The estimation of the maximum compression ratio per codec has been performed by
generated a number of test files using [gdal raster create](https://gdal.org/en/stable/programs/gdal_raster_create.html),
with 0-byte content, for different image sizes, and guessing the compression ratio pattern.

```console
gdal raster create --size 1024,1024 black_1024_packbits.tif       --co COMPRESS=PACKBITS --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_rle.tif       --co NBITS=1 --co COMPRESS=CCITTRLE --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_fax3.tif       --co NBITS=1 --co COMPRESS=CCITTFAX3 --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_fax4.tif       --co NBITS=1 --co COMPRESS=CCITTFAX4 --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_lzw.tif       --co COMPRESS=LZW --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_deflate.tif   --co COMPRESS=DEFLATE --co ZLEVEL=12 --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_zstd.tif   --co COMPRESS=ZSTD  --co ZSTD_LEVEL=22  --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_lzma.tif   --co COMPRESS=LZMA  --co LZMA_PRESET=9  --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_lerc.tif   --co COMPRESS=LERC --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_lerc_zstd.tif   --co COMPRESS=LERC_ZSTD  --co ZSTD_LEVEL=22  --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_lerc_deflate.tif   --co COMPRESS=LERC_DEFLATE  --co ZLEVEL=12  --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_jpg.tif       --co JPEG_QUALITY=1  --co COMPRESS=JPEG --co BLOCKYSIZE=1024 --overwrite
gdal raster create --band-count=3 --size 1024,1024 black_1024_rgb_jpg.tif   --co JPEG_QUALITY=1  --co COMPRESS=JPEG --co BLOCKYSIZE=1024 --overwrite
gdal raster create --band-count=3 --size 1024,1024 black_1024_ycbcr_jpg.tif   --co JPEG_QUALITY=1  --co COMPRESS=JPEG --co BLOCKYSIZE=1024 --co PHOTOMETRIC=YCBCR --overwrite
gdal raster create --band-count 3 --size 1024,1024 black_1024_rgb_webp.tif   --co COMPRESS=WEBP --co WEBP_LEVEL=1  --co BLOCKYSIZE=1024 --overwrite
gdal raster create --size 1024,1024 black_1024_jxl.tif   --co COMPRESS=JXL --co JXL_LOSSLESS=NO  --co JXL_DISTANCE=15 --co JXL_EFFORT=5  --co BLOCKYSIZE=1024 --overwrite
gdal raster create --band-count 3 --size 1024,1024 black_1024_rgb_jxl.tif   --co COMPRESS=JXL --co JXL_LOSSLESS=NO  --co JXL_DISTANCE=15 --co JXL_EFFORT=5  --co BLOCKYSIZE=1024 --overwrite

gdal raster create --size 4096,4096 black_4096_packbits.tif       --co COMPRESS=PACKBITS --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_rle.tif       --co NBITS=1 --co COMPRESS=CCITTRLE --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_fax3.tif       --co NBITS=1 --co COMPRESS=CCITTFAX3 --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_fax4.tif       --co NBITS=1 --co COMPRESS=CCITTFAX4 --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_lzw.tif       --co COMPRESS=LZW --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_deflate.tif   --co COMPRESS=DEFLATE --co ZLEVEL=12 --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_zstd.tif   --co COMPRESS=ZSTD  --co ZSTD_LEVEL=22  --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_lzma.tif   --co COMPRESS=LZMA  --co LZMA_PRESET=9  --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_lerc.tif   --co COMPRESS=LERC --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_lerc_zstd.tif   --co COMPRESS=LERC_ZSTD  --co ZSTD_LEVEL=22  --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_lerc_deflate.tif   --co COMPRESS=LERC_DEFLATE  --co ZLEVEL=12  --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_jpg.tif       --co JPEG_QUALITY=1  --co COMPRESS=JPEG --co BLOCKYSIZE=4096 --overwrite
gdal raster create --band-count=3 --size 4096,4096 black_4096_rgb_jpg.tif   --co JPEG_QUALITY=1  --co COMPRESS=JPEG --co BLOCKYSIZE=4096 --overwrite
gdal raster create --band-count=3 --size 4096,4096 black_4096_ycbcr_jpg.tif   --co JPEG_QUALITY=1  --co COMPRESS=JPEG --co BLOCKYSIZE=4096 --co PHOTOMETRIC=YCBCR --overwrite
gdal raster create --band-count 3 --size 4096,4096 black_4096_rgb_webp.tif   --co COMPRESS=WEBP --co WEBP_LEVEL=1  --co BLOCKYSIZE=4096 --overwrite
gdal raster create --size 4096,4096 black_4096_jxl.tif   --co COMPRESS=JXL --co JXL_LOSSLESS=NO  --co JXL_DISTANCE=15 --co JXL_EFFORT=5  --co BLOCKYSIZE=4096 --overwrite
gdal raster create --band-count 3 --size 4096,4096 black_4096_rgb_jxl.tif   --co COMPRESS=JXL --co JXL_LOSSLESS=NO  --co JXL_DISTANCE=15 --co JXL_EFFORT=5  --co BLOCKYSIZE=4096 --overwrite

gdal raster create --size 16383,16383 black_16383_packbits.tif       --co COMPRESS=PACKBITS --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_rle.tif       --co NBITS=1 --co COMPRESS=CCITTRLE --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_fax3.tif       --co NBITS=1 --co COMPRESS=CCITTFAX3 --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_fax4.tif       --co NBITS=1 --co COMPRESS=CCITTFAX4 --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_lzw.tif       --co COMPRESS=LZW --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_deflate.tif   --co COMPRESS=DEFLATE --co ZLEVEL=12 --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_zstd.tif   --co COMPRESS=ZSTD  --co ZSTD_LEVEL=22  --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_lzma.tif   --co COMPRESS=LZMA  --co LZMA_PRESET=9  --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_lerc.tif   --co COMPRESS=LERC --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_lerc_zstd.tif   --co COMPRESS=LERC_ZSTD  --co ZSTD_LEVEL=22  --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_lerc_deflate.tif   --co COMPRESS=LERC_DEFLATE  --co ZLEVEL=12  --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_jpg.tif       --co JPEG_QUALITY=1  --co COMPRESS=JPEG --co BLOCKYSIZE=16383 --overwrite
gdal raster create --band-count=3 --size 16383,16383 black_16383_rgb_jpg.tif   --co JPEG_QUALITY=1  --co COMPRESS=JPEG --co BLOCKYSIZE=16383 --overwrite
gdal raster create --band-count=3 --size 16383,16383 black_16383_ycbcr_jpg.tif   --co JPEG_QUALITY=1  --co COMPRESS=JPEG --co BLOCKYSIZE=16383 --co PHOTOMETRIC=YCBCR --overwrite
gdal raster create --band-count 3 --size 16383,16383 black_16383_rgb_webp.tif   --co COMPRESS=WEBP --co WEBP_LEVEL=1  --co BLOCKYSIZE=16383 --overwrite
gdal raster create --band-count 4 --size 16383,16383 black_16383_rgba_webp.tif   --co COMPRESS=WEBP --co WEBP_LEVEL=1  --co BLOCKYSIZE=16383 --overwrite
gdal raster create --size 16383,16383 black_16383_jxl.tif   --co COMPRESS=JXL --co JXL_LOSSLESS=NO  --co JXL_DISTANCE=15 --co JXL_EFFORT=5  --co BLOCKYSIZE=16383 --overwrite
gdal raster create --band-count 3 --size 16383,16383 black_16383_rgb_jxl.tif   --co COMPRESS=JXL --co JXL_LOSSLESS=NO  --co JXL_DISTANCE=15 --co JXL_EFFORT=5  --co BLOCKYSIZE=16383 --overwrite

gdal raster create --size 65536,65536 black_65536_rle.tif       --co NBITS=1 --co COMPRESS=CCITTRLE --co BLOCKYSIZE=65536 --overwrite
gdal raster create --size 65536,65536 black_65536_fax3.tif       --co NBITS=1 --co COMPRESS=CCITTFAX3 --co BLOCKYSIZE=65536 --overwrite
gdal raster create --size 65536,65536 black_65536_fax4.tif       --co NBITS=1 --co COMPRESS=CCITTFAX4 --co BLOCKYSIZE=65536 --overwrite
gdal raster create --size 65536,65536 black_65536_deflate.tif   --co COMPRESS=DEFLATE --co ZLEVEL=12 --co BLOCKYSIZE=65536 --overwrite
gdal raster create --size 65536,65536 black_65536_zstd.tif   --co COMPRESS=ZSTD  --co ZSTD_LEVEL=22  --co BLOCKYSIZE=65536 --overwrite
gdal raster create --size 65536,65536 black_65536_lzma.tif   --co COMPRESS=LZMA  --co LZMA_PRESET=9  --co BLOCKYSIZE=65536 --overwrite
gdal raster create --size 65536,65536 black_65536_lzw.tif   --co COMPRESS=LZW  --co BLOCKYSIZE=65536 --overwrite
```

and then extracted the compression ratios with:

```console
for i in black*.tif; do bytecounts=$(tiffdump $i | grep StripByteCounts | sed "s/StripByteCounts (279) LONG (4) 1<//" | sed "s/>//"); python3 -c "import math; uncompressed_size = 1024 * 1024 if '1024' in '$i' else 4096 * 4096 if '4096' in '$i' else 65536 * 65536 if '65536' in '$i' else 200000 * 200000 if '200000' in '$i' else 16363 * 16383; uncompressed_size = uncompressed_size * 4 if 'rgba' in '$i' else uncompressed_size * 3 if 'rgb' in '$i' or 'ycbcr' in '$i' else uncompressed_size; uncompressed_size = uncompressed_size / 8 if 'rle' in '$i' or 'fax' in '$i' else uncompressed_size; print('$i', math.ceil(uncompressed_size / ${bytecounts}))"; done
```

which outputs

```console
black_1024_deflate.tif 975
black_1024_fax3.tif 36
black_1024_fax4.tif 1001
black_1024_jpg.tif 252
black_1024_jxl.tif 5323
black_1024_lerc_deflate.tif 23832
black_1024_lerc.tif 14980
black_1024_lerc_zstd.tif 18397
black_1024_lzma.tif 3800
black_1024_lzw.tif 562
black_1024_packbits.tif 64
black_1024_rgb_jpg.tif 255
black_1024_rgb_jxl.tif 16216
black_1024_rgb_webp.tif 1634
black_1024_rle.tif 43
black_1024_ycbcr_jpg.tif 502
black_1024_zstd.tif 21400
black_16383_deflate.tif 989
black_16383_fax3.tif 163
black_16383_fax4.tif 16339
black_16383_jpg.tif 256
black_16383_jxl.tif 45360
black_16383_lerc_deflate.tif 5703725
black_16383_lerc.tif 3829644
black_16383_lerc_zstd.tif 4394673
black_16383_lzma.tif 6846
black_16383_lzw.tif 1353
black_16383_packbits.tif 64
black_16383_rgba_webp_lossless.tif 104067
black_16383_rgba_webp.tif 2199
black_16383_rgb_jpg.tif 256
black_16383_rgb_jxl.tif 136148
black_16383_rgb_webp_lossless.tif 78050
black_16383_rgb_webp.tif 1685
black_16383_rle.tif 171
black_16383_ycbcr_jpg.tif 512
black_16383_zstd.tif 32669
black_4096_deflate.tif 989
black_4096_fax3.tif 100
black_4096_fax4.tif 4073
black_4096_jpg.tif 256
black_4096_jxl_rgba.tif 90934
black_4096_jxl.tif 25079
black_4096_lerc_deflate.tif 381301
black_4096_lerc.tif 239675
black_4096_lerc_zstd.tif 294338
black_4096_lzma.tif 6534
black_4096_lzw.tif 1243
black_4096_packbits.tif 64
black_4096_rgb_jpg.tif 256
black_4096_rgb_jxl.tif 75574
black_4096_rgb_webp.tif 1684
black_4096_rle.tif 128
black_4096_ycbcr_jpg.tif 512
black_4096_zstd.tif 31715
black_65536_deflate.tif 990
black_65536_fax3.tif 200
black_65536_fax4.tif 65513
black_65536_lzma.tif 6874
black_65536_lzw.tif 1362
black_65536_rle.tif 205
black_65536_zstd.tif 32764
```

