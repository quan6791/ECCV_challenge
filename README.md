# ECCV18_challenge
This is the submission code for ECCV challenges:

the challenge 02: Video decaption

the challange 03: Denoising and inpainting for fingerprint verification

The structure of challenge files:

       --challenge
       
              -- models (please, download models link for track-2 and put here
       
              -- source
               
                      --files
               
                      --predict.bash
       
              -- data_set
       
       
To run the code:
```sh
$ chmod +x run.bash
$ chmod +x predict.bash
$ ./run.bash
```
or
```sh
$./predict.bash
```

Pre-trained models links:

Track-2:
[stage1](https://www.dropbox.com/s/2kaa7ugradazx3c/stage01.h5?dl=0)
[stage2](https://www.dropbox.com/s/x5qhjf9l212n47h/stage02.h5?dl=0)


Track-3:
[stage1](https://github.com/quan6791/ECCV_challenge/blob/master/challenge03/model.npz)
[stage2](https://github.com/quan6791/ECCV_challenge/blob/master/challenge03/model_step02.npz)
[stage3](https://github.com/quan6791/ECCV_challenge/blob/master/challenge03/model_step03.npz)
[stage4](https://github.com/quan6791/ECCV_challenge/blob/master/challenge03/model_step04.npz)
[stage5](https://github.com/quan6791/ECCV_challenge/blob/master/challenge03/model_step05.npz)
[stage6](https://github.com/quan6791/ECCV_challenge/blob/master/challenge03/model_step06.npz)
[stage7](https://github.com/quan6791/ECCV_challenge/blob/master/challenge03/model_step07.npz)
[stage8](https://github.com/quan6791/ECCV_challenge/blob/master/challenge03/model_step08.npz)


Challenge references:

[Video decaption](https://competitions.codalab.org/competitions/18421)

[Denoising and inpainting for fingerprint verification](https://competitions.codalab.org/competitions/18426)
