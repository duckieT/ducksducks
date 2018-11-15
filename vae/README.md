# vae

how to run?

## tldr

with a new model:
```
./go <(./model --model naive) <(./task --train ~/data/train-set --test ~/data/test-set --out .)
```

resuming from a previous training task:
```
./go model_10.pt task_10.pt
```

## model

model will output a `.pt` file describing the model to stdout, which can be read by `./go`.
As `./go` updates the model, it will output more `.pt` files describing the updated model, in the same format.
```
./model --model naive
```

run `./model -h` to find out more options

## task

task will output a `.pt` file describing the task to stdout, which can be read by `./go`
As `./go` updates the task, it will output more `.pt` files describing the updated task, in the same format.
```
./task --train ~/data/train-set --test ~/data/test-set --out .
```

run `./task -h` to find out more options
