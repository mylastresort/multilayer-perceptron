# Changelog

## 1.0.0 (2026-05-26)


### Features

* add preprocessing scalers and tests ([9e9bb0c](https://github.com/mylastresort/multilayer-perceptron/commit/9e9bb0c24e64dc062615e30e657391672de52727))
* **app:** add split, train, and predict subcommands with model save/load ([67f12a0](https://github.com/mylastresort/multilayer-perceptron/commit/67f12a009df9ccd0341c2274275e0fe43a3bc8cf))
* **app:** add subcommands, optimizers, model persistence, and test coverage ([e733580](https://github.com/mylastresort/multilayer-perceptron/commit/e733580e280f3da42ab07c6ff6f93508a492ecf1))
* **app:** split main into modules and refresh usage docs ([5b11166](https://github.com/mylastresort/multilayer-perceptron/commit/5b111667553145e83e95eef2f93c57e073a28cec))
* **app:** wire training modules in entrypoint ([ce35ee0](https://github.com/mylastresort/multilayer-perceptron/commit/ce35ee0ad1f2a1d5219c83ead271c0189f4d661d))
* **config:** add YAML network config and runtime CLI overrides ([b631d5d](https://github.com/mylastresort/multilayer-perceptron/commit/b631d5db7788a3437b38a5d88d6b8c2cbad51fad))
* **data:** add csv loader with unit tests ([1a0eb61](https://github.com/mylastresort/multilayer-perceptron/commit/1a0eb61febe4f8af86a81537fe6b5b95f52f9c35))
* **data:** add reproducible stratified split with tests ([b24f102](https://github.com/mylastresort/multilayer-perceptron/commit/b24f102427286451be14691c3ac646c662369321))
* **deps:** add minifb x11 dev dependency ([d3cb0f1](https://github.com/mylastresort/multilayer-perceptron/commit/d3cb0f189eccccdff02c278285d5d7c20af8ad66))
* **network:** add activation module with inline ops and tests ([6547f14](https://github.com/mylastresort/multilayer-perceptron/commit/6547f14fe6c13d98a0a5cc0b53052eb7a418df01))
* **network:** add layer model and callbacks ([3c710b2](https://github.com/mylastresort/multilayer-perceptron/commit/3c710b2022ad9224831ab229569de9ee28428495))
* **network:** add model save and load for persistence ([53af67f](https://github.com/mylastresort/multilayer-perceptron/commit/53af67fda4a606cb9741deaff91d664244143336))
* subcommands, optimizers, and model persistence ([630190a](https://github.com/mylastresort/multilayer-perceptron/commit/630190a420ca7e46e86198cf7214af1df6ee02b3))
* **training:** add backprop batching and loss core ([9395c96](https://github.com/mylastresort/multilayer-perceptron/commit/9395c969415674c14e779fed0db390c06af8b9ff))
* **training:** add end-to-end training pipeline with visualization and tests ([e48e2d8](https://github.com/mylastresort/multilayer-perceptron/commit/e48e2d8dd3b5b089184a52f0b3e09e603207405a))
* **training:** add monitored metrics, early stopping, and GUI curves ([906a444](https://github.com/mylastresort/multilayer-perceptron/commit/906a444f6638cfc12521f3be706491c4802cff9d))
* **training:** add Nesterov momentum, RMSprop, and Adam optimizers ([d08083d](https://github.com/mylastresort/multilayer-perceptron/commit/d08083db678dc90e838ac573ffb4f6d40f00943f))
* **training:** add trainer loop and metrics ([fea3786](https://github.com/mylastresort/multilayer-perceptron/commit/fea3786d92d75a773b863fb4d63fa60c16775a24))
* **training:** add YAML config, monitoring, and modular CLI ([06f5998](https://github.com/mylastresort/multilayer-perceptron/commit/06f5998de8916bc0d3a87c2205ead26e2e1262ec))
* **visualization:** add loss curve plotting ([6959834](https://github.com/mylastresort/multilayer-perceptron/commit/6959834147bf69d97f2cc0d89133cc92e8e8b70f))


### Bug Fixes

* **ci:** use tarpaulin llvm engine to avoid polars-arrow const eval panic ([d63de17](https://github.com/mylastresort/multilayer-perceptron/commit/d63de1711630cd1c89c1397676ad6b926fcace88))
* **data:** make stratified split deterministic with seed ([a83ec12](https://github.com/mylastresort/multilayer-perceptron/commit/a83ec124d22af6d769fb85071baea9091e0da389))
* **network:** require at least 2 hidden layers in config validation ([83019d8](https://github.com/mylastresort/multilayer-perceptron/commit/83019d88d41601aadc2b65bbf2729444b65537ed))

## Changelog

All notable changes to this project will be documented here.

This file is **auto-generated** by [Release Please](https://github.com/googleapis/release-please)
on every merge to `main`. Do not edit manually.

<!-- RELEASE-PLEASE-START -->
<!-- RELEASE-PLEASE-END -->
