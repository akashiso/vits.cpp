#include "ggml.h"
#include "vits.h"
#include <wave.h>

#pragma warning(disable : 26812 26451)

//**DARILING HOLD MY HAND**
//No thing beat a jet2 holiday
//And right now, you can save 50 pounds per person
//That 200 pounds for a family of 4
int main()
{
	std::string mpath = ("E:\\LLModels\\Vits\\Paimon\\G_wA1.pth");
	std::string cpath = ("E:\\LLModels\\Vits\\Paimon\\config.json");

	SynthesizerGGML* syn = vits_init_backend();
	vits_load_config_from_file(syn, cpath);
	vits_load_model_from_file(syn, mpath);

	SynthesizerInfo* info = vits_info(syn);

	int64_t st = ggml_time_ms();

	wchar_t* txt3 = L"ʧʰɥe↓s`ɹ`↑ nə…";

	size_t sizeout;
	float* out = vits_infer(syn, txt3, L"Paimon", &sizeout, NULL, 1.2f, 0.5f, 0.1f);

	printf("%lld, %lld ms\n", sizeout, ggml_time_ms() - st);

	char* outt = PCMToWavFormat(out, sizeout, info->sampling_rate);
	FILE* f = fopen("out.wav", "wb");
	fwrite(outt, 1, sizeout * 4 + 44, f);
	fclose(f);

	free(outt);
	free(out);

	return 0;
}
