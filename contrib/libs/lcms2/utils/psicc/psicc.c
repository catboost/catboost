//---------------------------------------------------------------------------------
//
//  Little Color Management System
//  Copyright (c) 1998-2024 Marti Maria Saguer
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

#include "utils.h"

// ------------------------------------------------------------------------

static char *cInProf = NULL;
static char *cOutProf = NULL;
static int Intent = INTENT_PERCEPTUAL;
static FILE* OutFile;
static int BlackPointCompensation = FALSE;
static int Undecorated = FALSE;
static int PrecalcMode = 1;
static int NumOfGridPoints = 0;



static
void Help(void)
{
    fprintf(stderr, "usage: psicc [flags] [<Output file>]\n\n");

    fprintf(stderr, "flags:\n\n");

    fprintf(stderr, "-i<profile> - Input profile: Generates Color Space Array (CSA)\n");
    fprintf(stderr, "-o<profile> - Output profile: Generates Color Rendering Dictionary(CRD)\n");

    fprintf(stderr, "-t<0,1,2,3> - Intent (0=Perceptual, 1=Colorimetric, 2=Saturation, 3=Absolute)\n");

    fprintf(stderr, "-b - Black point compensation (CRD only)\n");
    fprintf(stderr, "-u - Do NOT generate resource name on CRD\n");
    fprintf(stderr, "-c<0,1,2> - Precision (0=LowRes, 1=Normal (default), 2=Hi-res) (CRD only)\n");
    fprintf(stderr, "-n<gridpoints> - Alternate way to set precision, number of CLUT points (CRD only)\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "If no output file is specified, output goes to stdout.\n\n");
    fprintf(stderr, "This program is intended to be a demo of the little cms\n"
        "engine. Both lcms and this program are freeware. You can\n"
        "obtain both in source code at https://www.littlecms.com\n"
        "For suggestions, comments, bug reports etc. send mail to\n"
        "info@littlecms.com\n\n");

    exit(0);
}

// The toggles stuff

static
void HandleSwitches(int argc, char *argv[])
{
       int s;
      
       while ((s = xgetopt(argc,argv,"uUbBI:i:O:o:T:t:c:C:n:N:-:")) != EOF) {

       switch (s) 
       {

       case '-':
           if (strcmp(xoptarg, "help") == 0)
           {
               Help();
           }
           else
           {
               FatalError("Unknown option - run without args to see valid ones.\n");
           }
           break;
	 
       case 'i':
       case 'I':
            cInProf = xoptarg;
            break;

       case 'o':
       case 'O':
           cOutProf = xoptarg;
            break;

       case 'b':
       case 'B': BlackPointCompensation =TRUE;
            break;


       case 't':
       case 'T':
            Intent = atoi(xoptarg);
            if (Intent > 3) Intent = 3;
            if (Intent < 0) Intent = 0;
            break;
     
       case 'U':
       case 'u':
            Undecorated = TRUE;
            break;

       case 'c':
       case 'C':
            PrecalcMode = atoi(xoptarg);
            if (PrecalcMode < 0 || PrecalcMode > 2)
                    FatalError("ERROR: Unknown precalc mode '%d'", PrecalcMode);
            break;


       case 'n':
       case 'N':
                if (PrecalcMode != 1)
                    FatalError("Precalc mode already specified");
                NumOfGridPoints = atoi(xoptarg);
                break;


  default:

       FatalError("Unknown option - run without args to see valid ones.\n");
    }       
    }
}




static
void GenerateCSA(void)
{
	cmsHPROFILE hProfile = OpenStockProfile(0, cInProf);
	size_t n;
	char* Buffer;

	if (hProfile == NULL) return;

	n = cmsGetPostScriptCSA(0, hProfile, Intent, 0, NULL, 0);
	if (n == 0) return;

    Buffer = (char*) malloc(n + 1);
    if (Buffer != NULL) {

        cmsGetPostScriptCSA(0, hProfile, Intent, 0, Buffer, (cmsUInt32Number) n);
        Buffer[n] = 0;

        fprintf(OutFile, "%s", Buffer);	

        free(Buffer);
    }

	cmsCloseProfile(hProfile);
}


static
void GenerateCRD(void)
{
	cmsHPROFILE hProfile = OpenStockProfile(0, cOutProf);
	size_t n;
	char* Buffer;
    cmsUInt32Number dwFlags = 0;
    
	if (hProfile == NULL) return;

    if (BlackPointCompensation) dwFlags |= cmsFLAGS_BLACKPOINTCOMPENSATION;
    if (Undecorated)            dwFlags |= cmsFLAGS_NODEFAULTRESOURCEDEF;

    switch (PrecalcMode) {
           	
	    case 0: dwFlags |= cmsFLAGS_LOWRESPRECALC; break;
		case 2: dwFlags |= cmsFLAGS_HIGHRESPRECALC; break;
		case 1: 
            if (NumOfGridPoints > 0)
                dwFlags |= cmsFLAGS_GRIDPOINTS(NumOfGridPoints);
            break;

		default: FatalError("ERROR: Unknown precalculation mode '%d'", PrecalcMode);
	 }

	n = cmsGetPostScriptCRD(0, hProfile, Intent, dwFlags, NULL, 0);
	if (n == 0) return;

	Buffer = (char*) malloc(n + 1);
	if (Buffer == NULL) return;
        cmsGetPostScriptCRD(0, hProfile, Intent, dwFlags, Buffer, (cmsUInt32Number) n);
	Buffer[n] = 0;

	fprintf(OutFile, "%s", Buffer);			
	free(Buffer);
	cmsCloseProfile(hProfile);
}


int main(int argc, char *argv[])
{
	int nargs;

    fprintf(stderr, "Little CMS ICC PostScript generator - v2.1 [LittleCMS %2.2f]\n", cmsGetEncodedCMMversion() / 1000.0);
    fprintf(stderr, "Copyright (c) 1998-2024 Marti Maria Saguer. See COPYING file for details.\n");
    fflush(stderr);

	// Initialize
	InitUtils("psicc");

	 HandleSwitches(argc, argv);

     nargs = (argc - xoptind);
	 if (nargs != 0 && nargs != 1)
				Help();            
	
    if (cInProf == NULL && cOutProf == NULL)
        Help();

	 if (nargs == 0) 
			OutFile = stdout;
	 else
			OutFile = fopen(argv[xoptind], "wt");
	   		
	  if (cInProf != NULL)
			GenerateCSA();
		  
	  if (cOutProf != NULL)
			GenerateCRD();
		
	  if (nargs == 1) {
		  fclose(OutFile);
	  }

      return 0;     
}


