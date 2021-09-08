#import <Accelerate/Accelerate.h>

extern "C"
{
void iossgemm(float* Ap, int AM, int AN,
			  float* Bp, int BM, int BN,
			  float* Cp, int CM, int CN,
			  int bs, bool transposeA, bool transposeB)
	{
		cblas_sgemm(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans,
					transposeB ? CblasTrans : CblasNoTrans,
					AM, BN, BM, 1.0f, Ap, AN, Bp, BN, 1.0f, Cp, CN);
	}

}
