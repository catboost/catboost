//---------------------------------------------------------------------------------
//
//  Little Color Management System
//  Copyright (c) 1998-2023 Marti Maria Saguer
//
// Permission is hereby granted, free of charge, to any person obtaining 
// a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation 
// the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software 
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO 
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//---------------------------------------------------------------------------------

// This program does apply profiles to (some) TIFF files

#include "lcms2_plugin.h"
#include "tiffio.h"
#include "utils.h"

// Fix broken libtiff 4.3.0, thanks to Bob Friesenhahn for uncovering this

#if defined(HAVE_STDINT_H) && (TIFFLIB_VERSION >= 20201219)
#include <stdint.h>
#  undef cmsUInt16Number
#  define cmsUInt16Number uint16_t
#  undef cmsUInt32Number
#  define cmsUInt32Number uint32_t
#endif /* TIFFLIB_VERSION */

// Flags

static cmsBool BlackWhiteCompensation = FALSE;
static cmsBool IgnoreEmbedded         = FALSE;
static cmsBool EmbedProfile           = FALSE;
static int     PixelDepth             = 8;
static cmsBool GamutCheck             = FALSE;
static cmsBool lIsDeviceLink          = FALSE;
static cmsBool lIsCUBE                = FALSE;
static cmsBool StoreAsAlpha           = FALSE;

static int Intent                  = INTENT_PERCEPTUAL;
static int ProofingIntent          = INTENT_PERCEPTUAL;
static int PrecalcMode             = 1;
static cmsFloat64Number InkLimit   = 400;

static cmsFloat64Number ObserverAdaptationState  = 1.0;  // According ICC 4.3 this is the default

static const char *cInpProf  = NULL;
static const char *cOutProf  = NULL;
static const char *cProofing = NULL;

static const char* SaveEmbedded = NULL;

// Console error & warning
static
void ConsoleWarningHandler(const char* module, const char* fmt, va_list ap)
{
    if (Verbose) {

        fprintf(stderr, "Warning: ");

        if (module != NULL)
            fprintf(stderr, "[%s] ", module);

        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
}

static
void ConsoleErrorHandler(const char* module, const char* fmt, va_list ap)
{
    if (Verbose) {
        
        fprintf(stderr, "Error: ");

        if (module != NULL)
            fprintf(stderr, "[%s] ", module);
      
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
        fflush(stderr);
    }

}


// Issue a warning
static
void Warning(const char *frm, ...)
{
    va_list args;

    va_start(args, frm);
    ConsoleWarningHandler("tificc", frm, args);
    va_end(args);
}



// Out of memory is a fatal error
static
void OutOfMem(cmsUInt32Number size)
{
    FatalError("Out of memory on allocating %d bytes.", size);  
}


// -----------------------------------------------------------------------------------------------
// Lab plug-in
// In TIFF, Lab is encoded in a different way, so let's use the plug-in 
// capabilities of lcms2 to change the meaning of TYPE_Lab_8.  

#define LABTIFF_SH(m)           ((m) << 30)
#define T_LABTIFF(m)            (((m)>>30)&1)

// * 0xffff / 0xff00 = (255 * 257) / (255 * 256) = 257 / 256
static 
int FromLabV2ToLabV4(int x) 
{
    int a;

    a = ((x << 8) | x) >> 8;  // * 257 / 256
    if ( a > 0xffff) return 0xffff;
    return a;
}

// * 0xf00 / 0xffff = * 256 / 257
static 
int FromLabV4ToLabV2(int x) 
{
    return ((x << 8) + 0x80) / 257;
}


// Formatter for 8bit Lab TIFF (photometric 8)
static
unsigned char* UnrollTIFFLab8(struct _cmstransform_struct* CMMcargo,
                              CMSREGISTER cmsUInt16Number wIn[], 
                              CMSREGISTER cmsUInt8Number* accum, 
                              CMSREGISTER cmsUInt32Number Stride)
{
    wIn[0] = (cmsUInt16Number) FromLabV2ToLabV4((accum[0]) << 8);
    wIn[1] = (cmsUInt16Number) FromLabV2ToLabV4(((accum[1] > 127) ? (accum[1] - 128) : (accum[1] + 128)) << 8);
    wIn[2] = (cmsUInt16Number) FromLabV2ToLabV4(((accum[2] > 127) ? (accum[2] - 128) : (accum[2] + 128)) << 8);

    return accum + 3;

    UTILS_UNUSED_PARAMETER(Stride);
    UTILS_UNUSED_PARAMETER(CMMcargo);
}

// Formatter for 16bit Lab TIFF (photometric 8)
static
unsigned char* UnrollTIFFLab16(struct _cmstransform_struct* CMMcargo,
                              CMSREGISTER cmsUInt16Number wIn[],
                              CMSREGISTER cmsUInt8Number* accum,
                              CMSREGISTER cmsUInt32Number Stride )
{
    cmsUInt16Number* accum16 = (cmsUInt16Number*) accum;

    wIn[0] = (cmsUInt16Number) FromLabV2ToLabV4(accum16[0]);
    wIn[1] = (cmsUInt16Number) FromLabV2ToLabV4(((accum16[1] > 0x7f00) ? (accum16[1] - 0x8000) : (accum16[1] + 0x8000)) );
    wIn[2] = (cmsUInt16Number) FromLabV2ToLabV4(((accum16[2] > 0x7f00) ? (accum16[2] - 0x8000) : (accum16[2] + 0x8000)) );

    return accum + 3 * sizeof(cmsUInt16Number);

    UTILS_UNUSED_PARAMETER(Stride);
    UTILS_UNUSED_PARAMETER(CMMcargo);
}


static
unsigned char* PackTIFFLab8(struct _cmstransform_struct* CMMcargo, 
                            CMSREGISTER cmsUInt16Number wOut[], 
                            CMSREGISTER cmsUInt8Number* output, 
                            CMSREGISTER cmsUInt32Number Stride)
{
    int a, b;

    *output++ = (cmsUInt8Number) (FromLabV4ToLabV2(wOut[0] + 0x0080) >> 8);

    a = (FromLabV4ToLabV2(wOut[1]) + 0x0080) >> 8;
    b = (FromLabV4ToLabV2(wOut[2]) + 0x0080) >> 8;

    *output++ = (cmsUInt8Number) ((a < 128) ? (a + 128) : (a - 128));
    *output++ = (cmsUInt8Number) ((b < 128) ? (b + 128) : (b - 128));
    
    return output;

    UTILS_UNUSED_PARAMETER(Stride);
    UTILS_UNUSED_PARAMETER(CMMcargo);
}


static
unsigned char* PackTIFFLabA8(struct _cmstransform_struct* CMMcargo, 
                            CMSREGISTER cmsUInt16Number wOut[], 
                            CMSREGISTER cmsUInt8Number* output, 
                            CMSREGISTER cmsUInt32Number Stride)
{
    int a, b;

    *output++ = (cmsUInt8Number) (FromLabV4ToLabV2(wOut[0] + 0x0080) >> 8);

    a = (FromLabV4ToLabV2(wOut[1]) + 0x0080) >> 8;
    b = (FromLabV4ToLabV2(wOut[2]) + 0x0080) >> 8;

    *output++ = (cmsUInt8Number) ((a < 128) ? (a + 128) : (a - 128));
    *output++ = (cmsUInt8Number) ((b < 128) ? (b + 128) : (b - 128));

    output++; // Alpha

    return output;

    UTILS_UNUSED_PARAMETER(Stride);
    UTILS_UNUSED_PARAMETER(CMMcargo);
}


static
unsigned char* PackTIFFLab16(struct _cmstransform_struct* CMMcargo, 
                            CMSREGISTER cmsUInt16Number wOut[], 
                            CMSREGISTER cmsUInt8Number* output, 
                            CMSREGISTER cmsUInt32Number Stride)
{
    int a, b;
    cmsUInt16Number* output16 = (cmsUInt16Number*) output;

    *output16++ = (cmsUInt16Number) FromLabV4ToLabV2(wOut[0]);

    a = FromLabV4ToLabV2(wOut[1]);
    b = FromLabV4ToLabV2(wOut[2]);

    *output16++ = (cmsUInt16Number) ((a < 0x7f00) ? (a + 0x8000) : (a - 0x8000));
    *output16++ = (cmsUInt16Number) ((b < 0x7f00) ? (b + 0x8000) : (b - 0x8000));
    
    return (cmsUInt8Number*) output16;

    UTILS_UNUSED_PARAMETER(Stride);
    UTILS_UNUSED_PARAMETER(CMMcargo);
}

static
unsigned char* PackTIFFLabA16(struct _cmstransform_struct* CMMcargo, 
                            CMSREGISTER cmsUInt16Number wOut[], 
                            CMSREGISTER cmsUInt8Number* output, 
                            CMSREGISTER cmsUInt32Number Stride)
{
    int a, b;
    cmsUInt16Number* output16 = (cmsUInt16Number*) output;

    *output16++ = (cmsUInt16Number) FromLabV4ToLabV2(wOut[0]);

    a = FromLabV4ToLabV2(wOut[1]);
    b = FromLabV4ToLabV2(wOut[2]);

    *output16++ = (cmsUInt16Number) ((a < 0x7f00) ? (a + 0x8000) : (a - 0x8000));
    *output16++ = (cmsUInt16Number) ((b < 0x7f00) ? (b + 0x8000) : (b - 0x8000));

    output16++; // Alpha

    return (cmsUInt8Number*) output16;

    UTILS_UNUSED_PARAMETER(Stride);
    UTILS_UNUSED_PARAMETER(CMMcargo);
}


static
cmsFormatter TiffFormatterFactory(cmsUInt32Number Type,
                                  cmsFormatterDirection Dir,
                                  cmsUInt32Number dwFlags)
{
    cmsFormatter Result = { NULL };
    int bps           = T_BYTES(Type);
    int IsTiffSpecial = T_LABTIFF(Type);

    if (IsTiffSpecial && !(dwFlags & CMS_PACK_FLAGS_FLOAT))
    {
        if (Dir == cmsFormatterInput)
        {
            Result.Fmt16 = (bps == 1) ? UnrollTIFFLab8 : UnrollTIFFLab16;
        }
        else
        {
            if (T_EXTRA(Type) == 1)
                Result.Fmt16 = (bps == 1) ? PackTIFFLabA8 : PackTIFFLabA16;
            else
                if (T_EXTRA(Type) == 0)
                    Result.Fmt16 = (bps == 1) ? PackTIFFLab8 : PackTIFFLab16;
        }
    }

    return Result;
}

static cmsPluginFormatters TiffLabPlugin = { {cmsPluginMagicNumber, 2000, cmsPluginFormattersSig, NULL}, TiffFormatterFactory };

// -----------------------------------------------------------------------------------------------

// Build up the pixeltype descriptor
static
cmsUInt32Number GetInputPixelType(TIFF *Bank)
{
    cmsUInt16Number Photometric, bps, spp, extra, PlanarConfig, *info;
    cmsUInt16Number Compression;
    int ColorChannels;
    int IsPremul = FALSE, IsPlanar = FALSE, IsFlt = FALSE, IsReverse = FALSE;
    int labTiffSpecial = FALSE;
    int pt = PT_ANY;
    
    TIFFGetFieldDefaulted(Bank,  TIFFTAG_BITSPERSAMPLE, &bps);

    if (bps == 1)
        FatalError("Sorry, bilevel TIFFs has nothing to do with ICC profiles");

    if (bps != 8 && bps != 16 && bps != 32)
        FatalError("Sorry, 8, 16 or 32 bits per sample only");

   
    TIFFGetFieldDefaulted(Bank, TIFFTAG_PLANARCONFIG, &PlanarConfig);

    switch (PlanarConfig) {

     case PLANARCONFIG_CONTIG: IsPlanar = 0; break;
     case PLANARCONFIG_SEPARATE: IsPlanar = 1; break;
     default:

         FatalError("Unsupported planar configuration (=%d) ", (int) PlanarConfig);
    }

    TIFFGetFieldDefaulted(Bank, TIFFTAG_SAMPLESPERPIXEL, &spp);

    // If Samples per pixel == 1, PlanarConfiguration is irrelevant and need not to be included.
    if (spp == 1) IsPlanar = 0;

    // Any alpha?
    TIFFGetFieldDefaulted(Bank, TIFFTAG_EXTRASAMPLES, &extra, &info);

    // Read alpha channels as colorant?
    if (StoreAsAlpha) {

        ColorChannels = spp;
        extra = 0;
    }
    else
        ColorChannels = spp - extra;

    // Is alpha premultiplied ? 
    IsPremul = ((extra == 1) && (info[0] == EXTRASAMPLE_ASSOCALPHA));


    // Get photometric interpretation and proceed accordly
    TIFFGetField(Bank, TIFFTAG_PHOTOMETRIC, &Photometric);

    switch (Photometric) {

    case PHOTOMETRIC_MINISWHITE:

        IsReverse = 1;

        // ... fall through ...

    case PHOTOMETRIC_MINISBLACK:                                   
        pt = PT_GRAY;                                
        break;

    case PHOTOMETRIC_RGB:                                   
        pt = PT_RGB;
        if (ColorChannels < 3)
            FatalError("Sorry, RGB needs at least 3 samples per pixel");
        break;

     case PHOTOMETRIC_PALETTE:                                             
         FatalError("Sorry, palette images not supported"); 
         break;

     case PHOTOMETRIC_SEPARATED: 
         pt = PixelTypeFromChanCount(ColorChannels);
         break;

     case PHOTOMETRIC_YCBCR:
         TIFFGetField(Bank, TIFFTAG_COMPRESSION, &Compression);
         {
             cmsUInt16Number subx, suby;

             pt = PT_YCbCr;
             TIFFGetFieldDefaulted(Bank, TIFFTAG_YCBCRSUBSAMPLING, &subx, &suby);
             if (subx != 1 || suby != 1)
                 FatalError("Sorry, subsampled images are not supported");
         }
         break;

     // Two Lab flavours
     case PHOTOMETRIC_ICCLAB:
         pt = PT_Lab;         
         break;

     case PHOTOMETRIC_CIELAB:
         pt = PT_Lab;
         labTiffSpecial = TRUE;
         break;

    // CIE Log2(L) (u',v') 
     case PHOTOMETRIC_LOGLUV:     

         TIFFSetField(Bank, TIFFTAG_SGILOGDATAFMT, SGILOGDATAFMT_16BIT);
         pt = PT_YUV;             // *ICCSpace = icSigLuvData;
         bps = 16;                // 16 bits forced by LibTiff
         break;

     default:
         FatalError("Unsupported TIFF color space (Photometric %d)", Photometric);
    }

    // Convert bits per sample to bytes per sample

    bps >>= 3; 
    IsFlt = (bps == 0) || (bps == 4);

    return (FLOAT_SH(IsFlt) | COLORSPACE_SH(pt) | PLANAR_SH(IsPlanar) | EXTRA_SH(extra) | PREMUL_SH(IsPremul) |
           CHANNELS_SH(ColorChannels) | BYTES_SH(bps) | FLAVOR_SH(IsReverse) | LABTIFF_SH(labTiffSpecial));
}


// Rearrange pixel type to build output descriptor

static
cmsUInt32Number ComputeOutputFormatDescriptor(cmsUInt32Number dwInput, int OutColorSpace, int bps)
{
    int IsPlanar  = T_PLANAR(dwInput);
    int Channels  = ChanCountFromPixelType(OutColorSpace);
    int IsFlt = (bps == 0) || (bps == 4);
    int labTiffSpecial = FALSE;
    int Extra = T_EXTRA(dwInput);
    int IsPremul = T_PREMUL(dwInput);
    
    if (OutColorSpace == PT_Lab)
        labTiffSpecial = TRUE;

    return (FLOAT_SH(IsFlt) | COLORSPACE_SH(OutColorSpace) | PLANAR_SH(IsPlanar) |
        CHANNELS_SH(Channels) | BYTES_SH(bps) | EXTRA_SH(Extra) | PREMUL_SH(IsPremul) |
        LABTIFF_SH(labTiffSpecial));
}



// Tile based transforms
static
int TileBasedXform(cmsHTRANSFORM hXForm, TIFF* in, TIFF* out, int nPlanes)
{
    tsize_t BufSizeIn  = TIFFTileSize(in);
    tsize_t BufSizeOut = TIFFTileSize(out);
    unsigned char *BufferIn, *BufferOut;
    ttile_t i, TileCount = TIFFNumberOfTiles(in) / nPlanes;
    cmsUInt32Number tw, tl;
    int PixelCount, j;

    // Check for bad tiffs
    if (BufSizeIn > INT_MAX || BufSizeOut > INT_MAX)
        FatalError("Probably corrupted TIFF, tile too big.");

    TIFFGetFieldDefaulted(in, TIFFTAG_TILEWIDTH,  &tw);
    TIFFGetFieldDefaulted(in, TIFFTAG_TILELENGTH, &tl);

    PixelCount = (int) tw * tl;

    BufferIn = (unsigned char *) _TIFFmalloc(BufSizeIn * nPlanes);
    if (!BufferIn) OutOfMem((cmsUInt32Number) BufSizeIn * nPlanes);

    BufferOut = (unsigned char *) _TIFFmalloc(BufSizeOut * nPlanes);
    if (!BufferOut) OutOfMem((cmsUInt32Number) BufSizeOut * nPlanes);


    for (i = 0; i < TileCount; i++) {

        for (j=0; j < nPlanes; j++) {

            if (TIFFReadEncodedTile(in, i + (j* TileCount), 
                BufferIn + (j*BufSizeIn), BufSizeIn) < 0)   goto cleanup;
        }

        if (PixelCount < 0)
            FatalError("TIFF is corrupted");

        cmsDoTransform(hXForm, BufferIn, BufferOut, PixelCount);

        for (j=0; j < nPlanes; j++) {

            if (TIFFWriteEncodedTile(out, i + (j*TileCount),
                BufferOut + (j*BufSizeOut), BufSizeOut) < 0) goto cleanup;
        }
    }

    _TIFFfree(BufferIn);
    _TIFFfree(BufferOut);
    return 1;


cleanup:

    _TIFFfree(BufferIn);
    _TIFFfree(BufferOut);
    return 0;
}


// Strip based transforms
static
int StripBasedXform(cmsHTRANSFORM hXForm, TIFF* in, TIFF* out, int nPlanes)
{
    tsize_t BufSizeIn  = TIFFStripSize(in);
    tsize_t BufSizeOut = TIFFStripSize(out);
    unsigned char *BufferIn, *BufferOut;
    ttile_t i, StripCount = TIFFNumberOfStrips(in) / nPlanes;
    cmsUInt32Number sw;
    cmsUInt32Number sl;
    cmsUInt32Number iml;
    int j;
    int PixelCount;

    // Check for bad tiffs
    if (BufSizeIn > INT_MAX || BufSizeOut > INT_MAX)
        FatalError("Probably corrupted TIFF, strip too big.");

    TIFFGetFieldDefaulted(in, TIFFTAG_IMAGEWIDTH,  &sw);
    TIFFGetFieldDefaulted(in, TIFFTAG_ROWSPERSTRIP, &sl);
    TIFFGetFieldDefaulted(in, TIFFTAG_IMAGELENGTH, &iml);

    // It is possible to get infinite rows per strip
    if (sl == 0 || sl > iml)
        sl = iml;   // One strip for whole image

    BufferIn = (unsigned char *) _TIFFmalloc(BufSizeIn * nPlanes);
    if (!BufferIn) OutOfMem((cmsUInt32Number) BufSizeIn * nPlanes);

    BufferOut = (unsigned char *) _TIFFmalloc(BufSizeOut * nPlanes);
    if (!BufferOut) OutOfMem((cmsUInt32Number) BufSizeOut * nPlanes);


    for (i = 0; i < StripCount; i++) {

        for (j=0; j < nPlanes; j++) {

            if (TIFFReadEncodedStrip(in, i + (j * StripCount), 
                BufferIn + (j * BufSizeIn), BufSizeIn) < 0)   goto cleanup;
        }

        PixelCount = (int) sw * (iml < sl ? iml : sl);
        iml -= sl;

        if (PixelCount < 0)
            FatalError("TIFF is corrupted");

        cmsDoTransform(hXForm, BufferIn, BufferOut, PixelCount);

        for (j=0; j < nPlanes; j++) {
            if (TIFFWriteEncodedStrip(out, i + (j * StripCount), 
                BufferOut + j * BufSizeOut, BufSizeOut) < 0) goto cleanup;
        }

    }

    _TIFFfree(BufferIn);
    _TIFFfree(BufferOut);
    return 1;

cleanup:

    _TIFFfree(BufferIn);
    _TIFFfree(BufferOut);
    return 0;
}


// Fill the array with a short value
static
void fillArray(cmsInt16Number array[],  cmsInt16Number val, cmsUInt32Number size)
{
    cmsUInt32Number i;

    size /= sizeof(cmsInt16Number);
    for (i = 0; i < size; i++)
        array[i] = val;

}

// Creates minimum required tags
static
void WriteOutputTags(TIFF* out, int Colorspace, int BytesPerSample, int AlphaChannels, int IsPremul)
{
    int BitsPerSample = (8 * BytesPerSample);
    int nChannels = ChanCountFromPixelType(Colorspace);

   
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, BitsPerSample);
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, nChannels + AlphaChannels);

    if (AlphaChannels > 0)
    {
        cmsInt16Number Extra[20];

        if (IsPremul)
            fillArray(Extra, EXTRASAMPLE_ASSOCALPHA, sizeof(Extra));
        else
            fillArray(Extra, EXTRASAMPLE_UNASSALPHA, sizeof(Extra));

        TIFFSetField(out, TIFFTAG_EXTRASAMPLES, AlphaChannels, Extra);
    }

    switch (Colorspace) {

    case PT_GRAY:
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

        break;

    case PT_RGB:
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        break;

    case PT_CMY:
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_SEPARATED);
        TIFFSetField(out, TIFFTAG_INKSET, INKSET_MULTIINK);
        break;

    case PT_CMYK:
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_SEPARATED);
        TIFFSetField(out, TIFFTAG_INKSET, INKSET_CMYK);
        break;

    case PT_Lab:
        if (BitsPerSample == 16)
            TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_CIELAB);
        else
            TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_CIELAB);
        break;


        // Multi-ink separations
    case PT_MCH2:
    case PT_MCH3:
    case PT_MCH4:
    case PT_MCH5:
    case PT_MCH6:
    case PT_MCH7:
    case PT_MCH8:
    case PT_MCH9:
    case PT_MCH10:
    case PT_MCH11:
    case PT_MCH12:
    case PT_MCH13:
    case PT_MCH14:
    case PT_MCH15:

        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_SEPARATED);

        if (StoreAsAlpha && nChannels >= 4) {

            cmsInt16Number Extra[20];

            fillArray(Extra, EXTRASAMPLE_UNASSALPHA, sizeof(Extra));

            // CMYK plus extra alpha
            TIFFSetField(out, TIFFTAG_EXTRASAMPLES, nChannels - 4, Extra);
            TIFFSetField(out, TIFFTAG_INKSET, INKSET_CMYK);
            TIFFSetField(out, TIFFTAG_NUMBEROFINKS, 4);
        }
        else {
            TIFFSetField(out, TIFFTAG_INKSET, INKSET_MULTIINK);
            TIFFSetField(out, TIFFTAG_NUMBEROFINKS, nChannels);
        }
        break;


    default:
        FatalError("Unsupported output colorspace");
    }

    if (PixelDepth == 32)
        TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, 
                          SAMPLEFORMAT_IEEEFP, 
                          SAMPLEFORMAT_IEEEFP, 
                          SAMPLEFORMAT_IEEEFP, 
                          SAMPLEFORMAT_IEEEFP);
}


// Copies a bunch of tags

static
void CopyOtherTags(TIFF* in, TIFF* out)
{
#define CopyField(tag, v) \
    if (TIFFGetField(in, tag, &v)) TIFFSetField(out, tag, v)


    short shortv;
    unsigned short compression;
    cmsUInt32Number ow, ol;
    cmsFloat32Number floatv;
    char *stringv;
    cmsUInt32Number longv;

    CopyField(TIFFTAG_SUBFILETYPE, longv);

    TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &ow);
    TIFFGetField(in, TIFFTAG_IMAGELENGTH, &ol);

    TIFFSetField(out, TIFFTAG_IMAGEWIDTH, ow);
    TIFFSetField(out, TIFFTAG_IMAGELENGTH, ol);

    CopyField(TIFFTAG_PLANARCONFIG, shortv);
    CopyField(TIFFTAG_COMPRESSION, compression);

    // This is tricky, libtiff would access predictor in a wrong way 
    // if the codec is none of those
    if (compression == COMPRESSION_LZW ||
        compression == 34925 /*COMPRESSION_LZMA*/ ||
        compression == COMPRESSION_PIXARLOG || 
        compression == COMPRESSION_DEFLATE ||
        compression == COMPRESSION_ADOBE_DEFLATE ||
        compression == 50000 /*COMPRESSION_ZSTD*/)
    {
        if (PixelDepth != 32)
            CopyField(TIFFTAG_PREDICTOR, shortv);
    }

    CopyField(TIFFTAG_THRESHHOLDING, shortv);
    CopyField(TIFFTAG_FILLORDER, shortv);
    CopyField(TIFFTAG_ORIENTATION, shortv);
    CopyField(TIFFTAG_MINSAMPLEVALUE, shortv);
    CopyField(TIFFTAG_MAXSAMPLEVALUE, shortv);
    CopyField(TIFFTAG_XRESOLUTION, floatv);
    CopyField(TIFFTAG_YRESOLUTION, floatv);
    CopyField(TIFFTAG_RESOLUTIONUNIT, shortv);
    CopyField(TIFFTAG_ROWSPERSTRIP, longv);
    CopyField(TIFFTAG_XPOSITION, floatv);
    CopyField(TIFFTAG_YPOSITION, floatv);
    CopyField(TIFFTAG_IMAGEDEPTH, longv);
    CopyField(TIFFTAG_TILEDEPTH, longv);

    CopyField(TIFFTAG_TILEWIDTH,  longv);
    CopyField(TIFFTAG_TILELENGTH, longv);

    CopyField(TIFFTAG_ARTIST, stringv);
    CopyField(TIFFTAG_IMAGEDESCRIPTION, stringv);
    CopyField(TIFFTAG_MAKE, stringv);
    CopyField(TIFFTAG_MODEL, stringv);

    CopyField(TIFFTAG_DATETIME, stringv);
    CopyField(TIFFTAG_HOSTCOMPUTER, stringv);
    CopyField(TIFFTAG_PAGENAME, stringv);
    CopyField(TIFFTAG_DOCUMENTNAME, stringv);

}


static
void DoEmbedProfile(TIFF* Out, const char* ProfileFile)
{
    FILE* f;
    cmsInt32Number size;
    cmsUInt32Number EmbedLen;
    cmsUInt8Number* EmbedBuffer;

    f = fopen(ProfileFile, "rb");
    if (f == NULL) return;

    size = cmsfilelength(f);
    if (size < 0) return;

    EmbedBuffer = (cmsUInt8Number*) malloc((size_t) size + 1);
    if (EmbedBuffer == NULL) { 
        OutOfMem(size+1);
        return;
    }

    EmbedLen = (cmsUInt32Number) fread(EmbedBuffer, 1, (size_t) size, f);

    if (EmbedLen != (cmsUInt32Number) size) 
        FatalError("Cannot read %ld bytes to %s", (long) size, ProfileFile);

    fclose(f);
    EmbedBuffer[EmbedLen] = 0;

    TIFFSetField(Out, TIFFTAG_ICCPROFILE, EmbedLen, EmbedBuffer);
    free(EmbedBuffer);
}


// Read or create a ICC profile from the TIFF data
static
cmsHPROFILE GetTIFFProfile(TIFF* in)
{    
    cmsCIExyYTRIPLE Primaries;
    cmsFloat32Number* chr;
    cmsCIExyY WhitePoint;
    cmsFloat32Number* wp;
    int i;       
    cmsToneCurve* Curve[3]; 
    cmsUInt16Number *gmr, *gmg, *gmb;
    cmsHPROFILE hProfile;
    cmsUInt32Number EmbedLen;
    cmsUInt8Number* EmbedBuffer;

    if (IgnoreEmbedded) return NULL;

    if (TIFFGetField(in, TIFFTAG_ICCPROFILE, &EmbedLen, &EmbedBuffer)) {

        hProfile = cmsOpenProfileFromMem(EmbedBuffer, EmbedLen);

        // Print description found in the profile
        if (Verbose && (hProfile != NULL)) {

            fprintf(stdout, "\n[Embedded profile]\n");
            PrintProfileInformation(hProfile);                       
            fflush(stdout);
        }

        if (hProfile != NULL && SaveEmbedded != NULL)
            SaveMemoryBlock(EmbedBuffer, EmbedLen, SaveEmbedded);

        if (hProfile) return hProfile;
    }

    // Try to see if "colorimetric" tiff
    if (TIFFGetField(in, TIFFTAG_PRIMARYCHROMATICITIES, &chr)) {

        Primaries.Red.x   =  chr[0];
        Primaries.Red.y   =  chr[1];
        Primaries.Green.x =  chr[2];
        Primaries.Green.y =  chr[3];
        Primaries.Blue.x  =  chr[4];
        Primaries.Blue.y  =  chr[5];

        Primaries.Red.Y = Primaries.Green.Y = Primaries.Blue.Y = 1.0;

        if (TIFFGetField(in, TIFFTAG_WHITEPOINT, &wp)) {

            WhitePoint.x = wp[0];
            WhitePoint.y = wp[1];
            WhitePoint.Y = 1.0;

            // Transfer function is a bit harder....
            TIFFGetFieldDefaulted(in, TIFFTAG_TRANSFERFUNCTION,
                &gmr, 
                &gmg,
                &gmb);

            Curve[0] = cmsBuildTabulatedToneCurve16(NULL, 256, gmr);
            Curve[1] = cmsBuildTabulatedToneCurve16(NULL, 256, gmg);
            Curve[2] = cmsBuildTabulatedToneCurve16(NULL, 256, gmb);

            hProfile = cmsCreateRGBProfileTHR(NULL, &WhitePoint, &Primaries, Curve);

            for (i=0; i < 3; i++)
                cmsFreeToneCurve(Curve[i]);

            if (Verbose) {
                fprintf(stdout, "\n[Colorimetric TIFF]\n");
            }


            return hProfile;
        }
    }

    return NULL;
}


// Transform one image
static
int TransformImage(TIFF* in, TIFF* out, const char *cDefInpProf)
{
    cmsHPROFILE hIn, hOut, hProof, hInkLimit = NULL;
    cmsHTRANSFORM xform;
    cmsUInt32Number wInput, wOutput;
    int OutputColorSpace;
    int BytesPerSample = PixelDepth / 8;
    cmsUInt32Number dwFlags;        
    int nPlanes;

    // Default options
    dwFlags = cmsFLAGS_COPY_ALPHA;

    // Observer adaptation state (only meaningful on absolute colorimetric intent)
    cmsSetAdaptationState(ObserverAdaptationState);

    if (EmbedProfile && cOutProf) 
        DoEmbedProfile(out, cOutProf);

    if (BlackWhiteCompensation) 
        dwFlags |= cmsFLAGS_BLACKPOINTCOMPENSATION;           

    switch (PrecalcMode) {

       case 0: dwFlags |= cmsFLAGS_NOOPTIMIZE; break;
       case 2: dwFlags |= cmsFLAGS_HIGHRESPRECALC; break;
       case 3: dwFlags |= cmsFLAGS_LOWRESPRECALC; break;
       case 1: break;

       default: FatalError("Unknown precalculation mode '%d'", PrecalcMode);
    }

    if (GamutCheck)
        dwFlags |= cmsFLAGS_GAMUTCHECK;

    hProof = NULL;
    hOut = NULL;

    if (lIsDeviceLink) {

        if (lIsCUBE)
            hIn = cmsCreateDeviceLinkFromCubeFile(cDefInpProf);
        else
            hIn = cmsOpenProfileFromFile(cDefInpProf, "r");                  
    }
    else {

        hIn =  GetTIFFProfile(in);

        if (hIn == NULL)                    
            hIn = OpenStockProfile(NULL, cDefInpProf);               

        hOut = OpenStockProfile(NULL, cOutProf);

        if (cProofing != NULL) {

            hProof = OpenStockProfile(NULL, cProofing);
            dwFlags |= cmsFLAGS_SOFTPROOFING;
        }
    }

    // Take input color space
    wInput = GetInputPixelType(in);

    // Assure both, input profile and input TIFF are on same colorspace
    if (_cmsLCMScolorSpace(cmsGetColorSpace(hIn)) != (int) T_COLORSPACE(wInput))
        FatalError("Input profile is not operating in proper color space (%d)", T_COLORSPACE(wInput));


    if (!lIsDeviceLink) 
        OutputColorSpace = _cmsLCMScolorSpace(cmsGetColorSpace(hOut));
    else 
        OutputColorSpace = _cmsLCMScolorSpace(cmsGetPCS(hIn));

    wOutput  = ComputeOutputFormatDescriptor(wInput, OutputColorSpace, BytesPerSample);

    WriteOutputTags(out, OutputColorSpace, BytesPerSample, T_EXTRA(wOutput), T_PREMUL(wOutput));
    CopyOtherTags(in, out);

    // Ink limit
    if (InkLimit != 400.0 && 
        (OutputColorSpace == PT_CMYK || OutputColorSpace == PT_CMY)) {

            cmsHPROFILE hProfiles[10];
            int nProfiles = 0;

            hInkLimit = cmsCreateInkLimitingDeviceLink(cmsGetColorSpace(hOut), InkLimit);

            hProfiles[nProfiles++] = hIn;
            if (hProof) {
                hProfiles[nProfiles++] = hProof;
                hProfiles[nProfiles++] = hProof;
            }

            hProfiles[nProfiles++] = hOut;
            hProfiles[nProfiles++] = hInkLimit;

            xform = cmsCreateMultiprofileTransform(hProfiles, nProfiles, 
                                                   wInput, wOutput, Intent, dwFlags);

    }
    else {

        xform = cmsCreateProofingTransform(hIn, wInput, 
                                           hOut, wOutput, 
                                           hProof, Intent, 
                                           ProofingIntent, 
                                           dwFlags);
    }

    cmsCloseProfile(hIn);
    cmsCloseProfile(hOut);

    if (hInkLimit) 
        cmsCloseProfile(hInkLimit);
    if (hProof) 
        cmsCloseProfile(hProof);

    if (xform == NULL) return 0;

    // Planar stuff
    if (T_PLANAR(wInput)) 
        nPlanes = T_CHANNELS(wInput) + T_EXTRA(wInput);
    else
        nPlanes = 1;


    // Handle tile by tile or strip by strip
    if (TIFFIsTiled(in)) {

        TileBasedXform(xform, in, out, nPlanes);
    }
    else {
        StripBasedXform(xform, in, out, nPlanes);
    }


    cmsDeleteTransform(xform);

    TIFFWriteDirectory(out);

    return 1;
}


// Print help
static
void Help(int level)
{
    UTILS_UNUSED_PARAMETER(level);

    fprintf(stderr, "Little CMS ICC profile applier for TIFF - v8.0 [LittleCMS %2.2f]\n", cmsGetEncodedCMMversion() / 1000.0);
    fprintf(stderr, "Copyright (c) 1998-2023 Marti Maria Saguer. See COPYING file for details.\n");
    fflush(stderr);

    fprintf(stderr, "usage: tificc [flags] input.tif output.tif\n");

    fprintf(stderr, "\nflags:\n\n");
    fprintf(stderr, "-v - Verbose\n");
    fprintf(stderr, "-i<profile> - Input profile (defaults to sRGB)\n");
    fprintf(stderr, "-o<profile> - Output profile (defaults to sRGB)\n");
    fprintf(stderr, "-l<profile> - Transform by device-link profile\n");
    fprintf(stderr, "-u<profile> - Transform by CUBE colormap\n");

    PrintBuiltins();

    PrintRenderingIntents();

    fprintf(stderr, "-b - Black point compensation\n");
    fprintf(stderr, "-d<0..1> - Observer adaptation state (abs.col. only)\n");

    fprintf(stderr, "-c<0,1,2,3> - Precalculates transform (0=Off, 1=Normal, 2=Hi-res, 3=LoRes)\n");
    fprintf(stderr, "\n");

    fprintf(stderr, "-w<8,16,32> - Output depth. Use 32 for floating-point\n\n");
    fprintf(stderr, "-a - Handle channels > 4 as alpha\n");

    fprintf(stderr, "-n - Ignore embedded profile on input\n");
    fprintf(stderr, "-e - Embed destination profile\n");
    fprintf(stderr, "-s<new profile> - Save embedded profile as <new profile>\n");
    fprintf(stderr, "\n");


    fprintf(stderr, "-p<profile> - Soft proof profile\n");
    fprintf(stderr, "-m<n> - Soft proof intent\n");
    fprintf(stderr, "\tThat is the intent used to translate the simulation to the output device.\n\tNote that the simulated intent is set by using -t not by -m\n");
    fprintf(stderr, "-g - Marks out-of-gamut colors on softproof\n");

    fprintf(stderr, "\n");

    fprintf(stderr, "-k<0..400> - Ink-limiting in %% (CMYK only)\n");
    fprintf(stderr, "\n");
    

    fprintf(stderr, "Examples:\n\n"
        "To color correct from scanner to sRGB:\n"
        "\ttificc -iscanner.icm in.tif out.tif\n"
        "To convert from monitor1 to monitor2:\n"
        "\ttificc -imon1.icm -omon2.icm in.tif out.tif\n"
        "To make a CMYK separation:\n"
        "\ttificc -oprinter.icm inrgb.tif outcmyk.tif\n"
        "To recover sRGB from a CMYK separation:\n"
        "\ttificc -iprinter.icm incmyk.tif outrgb.tif\n"
        "To soft-proof how behaves Probev1_ICCv4.icc on perceptual:\n"
        "\ttifficc -t0 -p Probev1_ICCv4.icc -m1 infile.tif out.tif\n"
        "To convert from CIELab TIFF to sRGB\n"
        "\ttificc -i*Lab in.tif out.tif\n\n");


    fprintf(stderr, "This program is intended to be a demo of the Little CMS\n"
        "color engine. Both lcms and this program are open source.\n"
        "You can obtain both in source code at https://www.littlecms.com\n"
        "For suggestions, comments, bug reports etc. send mail to\n"
        "info@littlecms.com\n\n");

    exit(0);
}


// The toggles stuff

static
void HandleSwitches(int argc, char *argv[])
{
    int s;

    while ((s=xgetopt(argc,argv,"aAeEbBw:W:nNvVGgh:H:i:I:o:O:P:p:t:T:c:C:l:L:u:U:M:m:K:k:S:s:D:d:-:")) != EOF) {

        switch (s) {


        case '-':
            if (strcmp(xoptarg, "help") == 0)
            {
                Help(0);
            }
            else
            {
                FatalError("Unknown option - run without args to see valid ones.\n");
            }
            break;

        case 'a':
        case 'A':
            StoreAsAlpha = TRUE;
            break;
        case 'b':
        case 'B':
            BlackWhiteCompensation = TRUE;
            break;

        case 'c':
        case 'C':
            PrecalcMode = atoi(xoptarg);
            if (PrecalcMode < 0 || PrecalcMode > 3)
                FatalError("Unknown precalc mode '%d'", PrecalcMode);
            break;

        case 'd':
        case 'D': ObserverAdaptationState = atof(xoptarg);
            if (ObserverAdaptationState < 0 || 
                ObserverAdaptationState > 1.0)
                Warning("Adaptation state should be 0..1");
            break;

        case 'e':
        case 'E':
            EmbedProfile = TRUE;
            break;

        case 'g':
        case 'G':
            GamutCheck = TRUE;
            break;

        case 'v':
        case 'V':
            Verbose = TRUE;
            break;

        case 'i':
        case 'I':
            if (lIsDeviceLink)
                FatalError("Device-link already specified"); 

            cInpProf = xoptarg;
            break;

        case 'o':
        case 'O':
            if (lIsDeviceLink)
                FatalError("Device-link already specified"); 

            cOutProf = xoptarg;
            break;

        case 'l':
        case 'L': 
            if (cInpProf != NULL || cOutProf != NULL) 
                FatalError("input/output profiles already specified");

            cInpProf = xoptarg;
            lIsDeviceLink = TRUE;
            lIsCUBE = FALSE;
            break;

        case 'u':
        case 'U':
            if (cInpProf != NULL || cOutProf != NULL)
                FatalError("input/output profiles already specified");

            cInpProf = xoptarg;
            lIsDeviceLink = TRUE;
            lIsCUBE = TRUE;
            break;

        case 'p':
        case 'P':
            cProofing = xoptarg;
            break;

        case 't':
        case 'T':
            Intent = atoi(xoptarg);
            break;

        case 'm':
        case 'M':
            ProofingIntent = atoi(xoptarg);
            break;

        case 'N':
        case 'n':
            IgnoreEmbedded = TRUE;
            break;

        case 'W':
        case 'w':
            PixelDepth = atoi(xoptarg);
            if (PixelDepth != 8 && PixelDepth != 16 && PixelDepth != 32)
                FatalError("Only 8, 16 and 32 bps are supported");
            break;

        case 'k':
        case 'K':
            InkLimit = atof(xoptarg);
            if (InkLimit < 0.0 || InkLimit > 400.0)
                FatalError("Ink limit must be 0%%..400%%");
            break;


        case 's':
        case 'S': SaveEmbedded = xoptarg;
            break;

        case 'H':
        case 'h':  {

            int a =  atoi(xoptarg);
            Help(a); 
            }
            break;

        default:

            FatalError("Unknown option - run without args to see valid ones");
        }

    }
}


// The main sink

int main(int argc, char* argv[])
{
    TIFF *in, *out;
   
    cmsPlugin(&TiffLabPlugin);

    InitUtils("tificc");

    HandleSwitches(argc, argv);

    if ((argc - xoptind) != 2) {

        Help(0);              
    }
   

    TIFFSetErrorHandler(ConsoleErrorHandler);
    TIFFSetWarningHandler(ConsoleWarningHandler);

    in = TIFFOpen(argv[xoptind], "r");
    if (in == NULL) FatalError("Unable to open '%s'", argv[xoptind]);

    out = TIFFOpen(argv[xoptind+1], "w");

    if (out == NULL) {

        TIFFClose(in);
        FatalError("Unable to write '%s'", argv[xoptind+1]);
    }

    do {

        TransformImage(in, out, cInpProf);


    } while (TIFFReadDirectory(in));


    if (Verbose) { fprintf(stdout, "\n"); fflush(stdout); }

    TIFFClose(in);
    TIFFClose(out);

    return 0;
}

