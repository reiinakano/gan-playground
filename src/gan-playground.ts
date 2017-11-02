/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import './ndarray-image-visualizer';
import './ndarray-logits-visualizer';
import './model-layer';

// tslint:disable-next-line:max-line-length
import {Array1D, Array3D, DataStats, FeedEntry, Graph, InCPUMemoryShuffledInputProviderBuilder, Initializer, InMemoryDataset, MetricReduction, MomentumOptimizer, SGDOptimizer, RMSPropOptimizer, AdagradOptimizer, NDArray, NDArrayMath, NDArrayMathCPU, NDArrayMathGPU, Optimizer, OnesInitializer, Scalar, Session, Tensor, util, VarianceScalingInitializer, xhr_dataset, XhrDataset, XhrDatasetConfig, ZerosInitializer} from 'deeplearn';
import {NDArrayImageVisualizer} from './ndarray-image-visualizer';
import {NDArrayLogitsVisualizer} from './ndarray-logits-visualizer';
import {PolymerElement, PolymerHTMLElement} from './polymer-spec';

import {LayerBuilder, LayerWeightsDict} from './layer_builder';
import {ModelLayer} from './model-layer';
import * as model_builder_util from './model_builder_util';
import {Normalization} from './tensorflow';
import {getRandomInputProvider} from './my_input_provider';
import {MyGraphRunner, MyGraphRunnerEventObserver} from './my_graph_runner';

const DATASETS_CONFIG_JSON = 'model-builder-datasets-config.json';

/** How often to evaluate the model against test data. */
const EVAL_INTERVAL_MS = 1500;
/** How often to compute the cost. Downloading the cost stalls the GPU. */
const COST_INTERVAL_MS = 500;
/** How many inference examples to show when evaluating accuracy. */
const INFERENCE_EXAMPLE_COUNT = 15;
const INFERENCE_IMAGE_SIZE_PX = 100;
/**
 * How often to show inference examples. This should be less often than
 * EVAL_INTERVAL_MS as we only show inference examples during an eval.
 */
const INFERENCE_EXAMPLE_INTERVAL_MS = 3000;

// Smoothing factor for the examples/s standalone text statistic.
const EXAMPLE_SEC_STAT_SMOOTHING_FACTOR = .7;

const TRAIN_TEST_RATIO = 5 / 6;

const IMAGE_DATA_INDEX = 0;
const LABEL_DATA_INDEX = 1;

// tslint:disable-next-line:variable-name
export let GANPlaygroundPolymer: new () => PolymerHTMLElement = PolymerElement({
  is: 'gan-playground',
  properties: {
    inputShapeDisplay: String,
    isValid: Boolean,
    inferencesPerSec: Number,
    inferenceDuration: Number,
    generationsPerSec: Number,
    generationDuration: Number,
    examplesTrained: Number,
    examplesPerSec: Number,
    totalTimeSec: String,
    applicationState: Number,
    modelInitialized: Boolean,
    showTrainStats: Boolean,
    datasetDownloaded: Boolean,
    datasetNames: Array,
    selectedDatasetName: String,
    modelNames: Array,
    genModelNames: Array,
    discSelectedOptimizerName: String,
    genSelectedOptimizerName: String,
    optimizerNames: Array,
    discLearningRate: Number,
    genLearningRate: Number,
    discMomentum: Number,
    genMomentum: Number,
    discNeedMomentum: Boolean,
    genNeedMomentum: Boolean,
    discGamma: Number,
    genGamma: Number,
    discNeedGamma: Boolean,
    genNeedGamma: Boolean,
    batchSize: Number,
    selectedModelName: String,
    genSelectedModelName: String,
    selectedNormalizationOption:
        {type: Number, value: Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE},
    // Stats
    showDatasetStats: Boolean,
    statsInputMin: Number,
    statsInputMax: Number,
    statsInputShapeDisplay: String,
    statsLabelShapeDisplay: String,
    statsExampleCount: Number,
  }
});

export enum ApplicationState {
  IDLE = 1,
  TRAINING = 2
}

export class GANPlayground extends GANPlaygroundPolymer {
  // Polymer properties.
  private isValid: boolean;
  private totalTimeSec: string;
  private applicationState: ApplicationState;
  private modelInitialized: boolean;
  private showTrainStats: boolean;
  private selectedNormalizationOption: number;

  // Datasets and models.
  private graphRunner: MyGraphRunner;
  private graph: Graph;
  private session: Session;
  private discOptimizer: Optimizer;
  private genOptimizer: Optimizer;
  private xTensor: Tensor;
  private labelTensor: Tensor;
  private costTensor: Tensor;
  private accuracyTensor: Tensor;
  private predictionTensor: Tensor;
  private discPredictionReal: Tensor;
  private discPredictionFake: Tensor;
  private discLoss: Tensor;
  private genLoss: Tensor;
  private generatedImage: Tensor;

  private datasetDownloaded: boolean;
  private datasetNames: string[];
  private selectedDatasetName: string;
  private modelNames: string[];
  private genModelNames: string[];
  private selectedModelName: string;
  private genSelectedModelName: string;
  private optimizerNames: string[];
  private discSelectedOptimizerName: string;
  private genSelectedOptimizerName: string;
  private loadedWeights: LayerWeightsDict[]|null;
  private dataSets: {[datasetName: string]: InMemoryDataset};
  private dataSet: InMemoryDataset;
  private xhrDatasetConfigs: {[datasetName: string]: XhrDatasetConfig};
  private datasetStats: DataStats[];
  private discLearningRate: number;
  private genLearningRate: number;
  private discMomentum: number;
  private genMomentum: number;
  private discNeedMomentum: boolean;
  private genNeedMomentum: boolean;
  private discGamma: number;
  private genGamma: number;
  private discNeedGamma: boolean;
  private genNeedGamma: boolean;
  private batchSize: number;

  // Stats.
  private showDatasetStats: boolean;
  private statsInputRange: string;
  private statsInputShapeDisplay: string;
  private statsLabelShapeDisplay: string;
  private statsExampleCount: number;

  // Charts.
  private costChart: Chart;
  private accuracyChart: Chart;
  private examplesPerSecChart: Chart;
  private costChartData: ChartPoint[];
  private accuracyChartData: ChartPoint[];
  private examplesPerSecChartData: ChartPoint[];

  private trainButton: HTMLButtonElement;

  // Visualizers.
  private inputNDArrayVisualizers: NDArrayImageVisualizer[];
  private outputNDArrayVisualizers: NDArrayLogitsVisualizer[];

  private inputShape: number[];
  private labelShape: number[];
  private randVectorShape: number[];
  private examplesPerSec: number;
  private examplesTrained: number;
  private inferencesPerSec: number;
  private inferenceDuration: number;
  private generationsPerSec: number;
  private generationDuration: number;

  private inputLayer: ModelLayer;
  private hiddenLayers: ModelLayer[];

  private layersContainer: HTMLDivElement;

  private math: NDArrayMath;
  // Keep one instance of each NDArrayMath so we don't create a user-initiated
  // number of NDArrayMathGPU's.
  private mathGPU: NDArrayMathGPU;
  private mathCPU: NDArrayMathCPU;

  ready() {
    this.mathGPU = new NDArrayMathGPU();
    this.mathCPU = new NDArrayMathCPU();
    this.math = this.mathGPU;

    const eventObserver: MyGraphRunnerEventObserver = {
      batchesTrainedCallback: (batchesTrained: number) =>
          this.displayBatchesTrained(batchesTrained),
      discCostCallback: (cost: Scalar) => this.displayCost(cost, 'disc'),
      genCostCallback: (cost: Scalar) => this.displayCost(cost, 'gen'),
      metricCallback: (metric: Scalar) => this.displayAccuracy(metric),
      inferenceExamplesCallback:
          (inputFeeds: FeedEntry[][], inferenceOutputs: NDArray[][]) =>
              this.displayInferenceExamplesOutput(inputFeeds, inferenceOutputs),
      //        console.log(inputFeeds, inferenceOutputs),
      inferenceExamplesPerSecCallback: (examplesPerSec: number) =>
          this.displayInferenceExamplesPerSec(examplesPerSec),
      trainExamplesPerSecCallback: (examplesPerSec: number) =>
          this.displayExamplesPerSec(examplesPerSec),
      totalTimeCallback: (totalTimeSec: number) => this.totalTimeSec =
          totalTimeSec.toFixed(1),
    };
    this.graphRunner = new MyGraphRunner(this.math, this.session, eventObserver);

    // Set up datasets.
    this.populateDatasets();
    // this.createModel();

    this.querySelector('#dataset-dropdown .dropdown-content')
        .addEventListener(
            // tslint:disable-next-line:no-any
            'iron-activate', (event: any) => {
              // Update the dataset.
              const datasetName = event.detail.selected;
              this.updateSelectedDataset(datasetName);

              // TODO(nsthorat): Remember the last model used for each dataset.
              this.removeAllLayers();
            });
    this.querySelector('#model-dropdown .dropdown-content')
        .addEventListener(
            // tslint:disable-next-line:no-any
            'iron-activate', (event: any) => {
              // Update the model.
              const modelName = event.detail.selected;
              this.updateSelectedModel(modelName, 'disc');
            });
    this.querySelector('#gen-model-dropdown .dropdown-content')
        .addEventListener(
            // tslint:disable-next-line:no-any
            'iron-activate', (event: any) => {
              // Update the model.
              const modelName = event.detail.selected;
              this.updateSelectedModel(modelName, 'gen');
            });

    {
      const normalizationDropdown =
          this.querySelector('#normalization-dropdown .dropdown-content');
      // tslint:disable-next-line:no-any
      normalizationDropdown.addEventListener('iron-activate', (event: any) => {
        const selectedNormalizationOption = event.detail.selected;
        this.applyNormalization(selectedNormalizationOption);
        this.setupDatasetStats();
      });
    }
    this.querySelector("#disc-optimizer-dropdown .dropdown-content")
        // tslint:disable-next-line:no-any
        .addEventListener('iron-activate', (event: any) => {
          // Activate, deactivate hyper parameter inputs.
          this.refreshHyperParamRequirements(event.detail.selected,
            'disc');
        });

    this.querySelector("#gen-optimizer-dropdown .dropdown-content")
        // tslint:disable-next-line:no-any
        .addEventListener('iron-activate', (event: any) => {
          // Activate, deactivate hyper parameter inputs.
          this.refreshHyperParamRequirements(event.detail.selected,
            'gen');
        });

    this.discLearningRate = 0.02;
    this.genLearningRate = 0.01;
    this.discMomentum = 0.1;
    this.genMomentum = 0.1;
    this.discNeedMomentum = true;
    this.genNeedMomentum = true;
    this.discGamma = 0.1;
    this.genGamma = 0.1;
    this.discNeedGamma = true;
    this.genNeedGamma = true;
    this.batchSize = 15;
    // Default optimizer is momentum
    this.discSelectedOptimizerName = "rmsprop";
    this.genSelectedOptimizerName = "rmsprop";
    this.optimizerNames = ["sgd", "momentum", "rmsprop", "adagrad"];

    this.applicationState = ApplicationState.IDLE;
    this.loadedWeights = null;
    this.modelInitialized = false;
    this.showTrainStats = false;
    this.showDatasetStats = false;

    const addButton = this.querySelector('#add-layer');
    addButton.addEventListener('click', () => this.addLayer('disc'));

    const genAddButton = this.querySelector('#gen-add-layer');
    genAddButton.addEventListener('click', () => this.addLayer('gen'));

    const downloadModelButton = this.querySelector('#download-model');
    downloadModelButton.addEventListener('click', () => this.downloadModel());
    const uploadModelButton = this.querySelector('#upload-model');
    uploadModelButton.addEventListener('click', () => this.uploadModel());
    this.setupUploadModelButton();

    const uploadWeightsButton = this.querySelector('#upload-weights');
    uploadWeightsButton.addEventListener('click', () => this.uploadWeights());
    this.setupUploadWeightsButton();

    const stopButton = this.querySelector('#stop');
    stopButton.addEventListener('click', () => {
      this.applicationState = ApplicationState.IDLE;
      this.graphRunner.stopTraining();
    });

    this.trainButton = this.querySelector('#train') as HTMLButtonElement;
    this.trainButton.addEventListener('click', () => {
      this.createModel();
      this.startTraining();
    });

    this.querySelector('#environment-toggle')
        .addEventListener('change', (event) => {
          this.math =
              // tslint:disable-next-line:no-any
              (event.target as any).active ? this.mathGPU : this.mathCPU;
          this.graphRunner.setMath(this.math);
        });

    this.discHiddenLayers = [];
    this.genHiddenLayers = [];
    this.examplesPerSec = 0;
    this.inferencesPerSec = 0;
    this.generationsPerSec = 0;
    this.randVectorShape = [100];
  }

  isTraining(applicationState: ApplicationState): boolean {
    return applicationState === ApplicationState.TRAINING;
  }

  isIdle(applicationState: ApplicationState): boolean {
    return applicationState === ApplicationState.IDLE;
  }



  private getTestData(): NDArray[][] {
    const data = this.dataSet.getData();
    if (data == null) {
      return null;
    }
    const [images, labels] = this.dataSet.getData() as [NDArray[], NDArray[]];

    const start = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(start), labels.slice(start)];
  }

  private getTrainingData(): NDArray[][] {
    const [images, labels] = this.dataSet.getData() as [NDArray[], NDArray[]];

    const end = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(0, end), labels.slice(0, end)];
  }

  private getData(): NDArray[][] {
    return this.dataSet.getData() as [NDArray[], NDArray[]];
  }

  private getImageDataOnly(): NDArray[] {
    const [images, labels] = this.dataSet.getData() as [NDArray[], NDArray[]];
    return images
  }

  private startInference() {
    const data = this.getImageDataOnly();
    if(data == null) {
      return;
    }
    if (this.isValid && (data != null)) {
      const shuffledInputProviderGenerator = 
          new InCPUMemoryShuffledInputProviderBuilder([data]);
      const [inputImageProvider] =
          shuffledInputProviderGenerator.getInputProviders();

      const oneInputProvider = {
        getNextCopy(math: NDArrayMath): NDArray {
          return Array1D.new([0, 1]);
        },
        disposeCopy(math: NDArrayMath, copy: NDArray) {
          copy.dispose();
        }
      }

      const zeroInputProvider = {
        getNextCopy(math: NDArrayMath): NDArray {
          return Array1D.new([1, 0]);
        },
        disposeCopy(math: NDArrayMath, copy: NDArray) {
          copy.dispose();
        }
      }

      const inferenceFeeds = [
        {tensor: this.xTensor, data: inputImageProvider},
        {tensor: this.randomTensor, data: getRandomInputProvider(this.randVectorShape)},
        {tensor: this.oneTensor, data: oneInputProvider},
        {tensor: this.zeroTensor, data: zeroInputProvider}
      ]

      this.graphRunner.infer(
        this.generatedImage, this.discPredictionFake, this.discPredictionReal,
        inferenceFeeds, INFERENCE_EXAMPLE_INTERVAL_MS, INFERENCE_EXAMPLE_COUNT
      );
    }
  }

  private resetHyperParamRequirements(which: string) {
    if (which === 'gen') {
      this.genNeedMomentum = false;
      this.genNeedGamma = false;
    } else {
      this.discNeedMomentum = false;
      this.discNeedGamma = false;
    }
  }

  /**
   * Set flag to disable input by optimizer selection.
   */
  private refreshHyperParamRequirements(optimizerName: string,
    which: string) {
    this.resetHyperParamRequirements(which);
    switch (optimizerName) {
      case "sgd": {
        // No additional hyper parameters
        break;
      }
      case "momentum": {
        if (which === 'gen') {
          this.genNeedMomentum = true;
        } else {
          this.discNeedMomentum = true;
        }
        break;
      }
      case "rmsprop": {
        if (which === 'gen') {
          this.genNeedMomentum = true;
          this.genNeedGamma = true;
        } else {
          this.discNeedMomentum = true;
          this.discNeedGamma = true;
        }
        break;
      }
      case "adagrad": {
        if (which === 'gen') {
          this.genNeedMomentum = true;
        } else {
          this.discNeedMomentum = true;
        }
        break;
      }
      default: {
        throw new Error(`Unknown optimizer`);
      }
    }
  }

  private createOptimizer(which: string) {
    if (which === 'gen') {
      var selectedOptimizerName = this.genSelectedOptimizerName;
      var learningRate = this.genLearningRate;
      var momentum = this.genMomentum;
      var gamma = this.genGamma;
      var varName = 'generator';
    } else {
      var selectedOptimizerName = this.discSelectedOptimizerName;
      var learningRate = this.discLearningRate;
      var momentum = this.discMomentum;
      var gamma = this.discGamma;
      var varName = 'discriminator';
    }
    switch (selectedOptimizerName) {
      case 'sgd': {
        return new SGDOptimizer(+learningRate,
          this.graph.getNodes().filter((x) => 
            x.name.startsWith(varName)));
      }
      case 'momentum': {
        return new MomentumOptimizer(+learningRate, +momentum,
          this.graph.getNodes().filter((x) => 
            x.name.startsWith(varName)));
      }
      case 'rmsprop': {
        return new RMSPropOptimizer(+learningRate, +gamma,
          this.graph.getNodes().filter((x) => 
            x.name.startsWith(varName)));
      }
      case 'adagrad': {
        return new AdagradOptimizer(+learningRate, +gamma,
          this.graph.getNodes().filter((x) => 
            x.name.startsWith(varName)));
      }
      default: {
        throw new Error(`Unknown optimizer`);
      }
    }
  }

  private startTraining() {
    const data = this.getImageDataOnly();

    // Recreate optimizer with the selected optimizer and hyperparameters.
    this.discOptimizer = this.createOptimizer('disc');
    this.genOptimizer = this.createOptimizer('gen');

    if (this.isValid && data != null) {
      this.recreateCharts();
      this.graphRunner.resetStatistics();

      const shuffledInputProviderGenerator =
          new InCPUMemoryShuffledInputProviderBuilder([data]);
      const [inputImageProvider] =
          shuffledInputProviderGenerator.getInputProviders();

      const oneInputProvider = {
        getNextCopy(math: NDArrayMath): NDArray {
          return Array1D.new([0, 1]);
        },
        disposeCopy(math: NDArrayMath, copy: NDArray) {
          copy.dispose();
        }
      }

      const zeroInputProvider = {
        getNextCopy(math: NDArrayMath): NDArray {
          return Array1D.new([1, 0]);
        },
        disposeCopy(math: NDArrayMath, copy: NDArray) {
          copy.dispose();
        }
      }

      const discFeeds = [
        {tensor: this.xTensor, data: inputImageProvider},
        {tensor: this.randomTensor, data: getRandomInputProvider(this.randVectorShape)},
        {tensor: this.oneTensor, data: oneInputProvider},
        {tensor: this.zeroTensor, data: zeroInputProvider}
      ]

      const genFeeds = [
        {tensor: this.randomTensor, data: getRandomInputProvider(this.randVectorShape)},
        {tensor: this.oneTensor, data: oneInputProvider},
        {tensor: this.zeroTensor, data: zeroInputProvider}
      ]

      this.graphRunner.train(
        this.discLoss, this.genLoss, discFeeds, genFeeds, this.batchSize,
        this.discOptimizer, this.genOptimizer, undefined, COST_INTERVAL_MS);

      this.showTrainStats = true;
      this.applicationState = ApplicationState.TRAINING;
    }
  }

  /*
  private createModel() {
    if (this.session != null) {
      this.session.dispose();
    }

    this.modelInitialized = false;
    if (this.isValid === false) {
      return;
    }

    this.graph = new Graph();
    const g = this.graph;
    this.xTensor = g.placeholder('input', this.inputShape);
    this.labelTensor = g.placeholder('label', this.labelShape);

    let network = this.xTensor;

    for (let i = 0; i < this.discHiddenLayers.length; i++) {
      let weights: LayerWeightsDict|null = null;
      if (this.loadedWeights != null) {
        weights = this.loadedWeights[i];
      }
      network = this.discHiddenLayers[i].addLayer(g, network, i, weights);
    }
    this.predictionTensor = network;
    this.costTensor =
        g.softmaxCrossEntropyCost(this.predictionTensor, this.labelTensor);
    this.accuracyTensor =
        g.argmaxEquals(this.predictionTensor, this.labelTensor);

    this.loadedWeights = null;

    this.session = new Session(g, this.math);
    this.graphRunner.setSession(this.session);

    this.startInference();

    this.modelInitialized = true;
  }
  */

  private createModel() {
    if (this.session != null) {
      this.session.dispose();
    }

    this.modelInitialized = false;
    if (this.isValid === false) {
      return;
    }

    // Construct graph
    this.graph = new Graph();
    const g = this.graph;
    this.randomTensor = g.placeholder('random', this.randVectorShape);
    this.xTensor = g.placeholder('input', [28, 28, 1]);
    this.oneTensor = g.placeholder('one', [2]);
    this.zeroTensor = g.placeholder('zero', [2]);

    const varianceInitializer: Initializer = new VarianceScalingInitializer()
    const zerosInitializer: Initializer = new ZerosInitializer()
    const onesInitializer: Initializer = new OnesInitializer();

    // Construct generator
    let gen = this.randomTensor;
    for (let i = 0; i < this.genHiddenLayers.length; i++) {
      let weights: LayerWeightsDict|null = null;
      if (this.loadedWeights != null) {
        weights = this.loadedWeights[i];
      }
      [gen] = this.genHiddenLayers[i].addLayerMultiple(g, [gen], 
        'generator', weights);
    }
    gen = g.tanh(gen);

    /*
    const genHidden1Weight = g.variable(
      'generator-hidden-1-weight',
      varianceInitializer.initialize([100, 256], 100, 256)
    );
    gen = g.matmul(gen, genHidden1Weight);
    const genHidden1Bias = g.variable(
      'generator-hidden-1-bias',
      zerosInitializer.initialize([256], 100, 256)
    );
    gen = g.add(gen, genHidden1Bias)
    gen = g.relu(gen);
    const genOutWeight = g.variable(
      'generator-out-weight',
      varianceInitializer.initialize([256, 784], 256, 784)
    );
    gen = g.matmul(gen, genOutWeight);
    const genOutBias = g.variable(
      'generator-out-bias',
      zerosInitializer.initialize([784], 256, 784)
    );
    gen = g.add(gen, genOutBias);
    gen = g.reshape(gen, this.xTensor.shape);
    gen = g.tanh(gen);*/

    // Construct discriminator
    let disc1 = gen;
    let disc2 = this.xTensor;
    for (let i = 0; i < this.discHiddenLayers.length; i++) {
      let weights: LayerWeightsDict|null = null;
      if (this.loadedWeights != null) {
        weights = this.loadedWeights[i];
      }
      [disc1, disc2] = this.discHiddenLayers[i].addLayerMultiple(g, [disc1, disc2], 
        'discriminator', weights);
    }
    /*
    // Construct discriminator for generated images
    let disc1 = gen;
    disc1 = g.reshape(disc1, [disc1.shape[0]*disc1.shape[1]*disc1.shape[2]]);
    const discHidden1Weight = g.variable(
      'discriminator-hidden-1-weight',
      varianceInitializer.initialize([784, 256], 784, 256)
    );
    disc1 = g.matmul(disc1, discHidden1Weight);
    const discHidden1Bias = g.variable(
      'discriminator-hidden-1-bias',
      zerosInitializer.initialize([256], 784, 256)
    );
    disc1 = g.add(disc1, discHidden1Bias);
    disc1 = g.relu(disc1);
    const discOutWeight = g.variable(
      'discriminator-out-weight',
      varianceInitializer.initialize([256, 2], 256, 2)
    );
    disc1 = g.matmul(disc1, discOutWeight);
    const discOutBias = g.variable(
      'discriminator-out-bias',
      zerosInitializer.initialize([2], 256, 2)
    );
    disc1 = g.add(disc1, discOutBias);
    disc1 = g.sigmoid(disc1);

    // Construct second discriminator (sharing variables) for real images
    let disc2 = this.xTensor;
    disc2 = g.reshape(disc2, [disc2.shape[0]*disc2.shape[1]*disc2.shape[2]]);
    disc2 = g.matmul(disc2, discHidden1Weight);
    disc2 = g.add(disc2, discHidden1Bias);
    disc2 = g.relu(disc2);
    disc2 = g.matmul(disc2, discOutWeight);
    disc2 = g.add(disc2, discOutBias);
    disc2 = g.sigmoid(disc2);*/

    this.discPredictionReal = disc2;
    this.discPredictionFake = disc1;
    this.generatedImage = gen;
    const discLossReal = g.softmaxCrossEntropyCost(
      this.discPredictionReal,
      this.oneTensor
    );
    const discLossFake = g.softmaxCrossEntropyCost(
      this.discPredictionFake,
      this.zeroTensor
    );
    this.discLoss = g.add(discLossReal, discLossFake);

    this.genLoss = g.softmaxCrossEntropyCost(
      this.discPredictionFake,
      this.oneTensor
    );

    this.session = new Session(g, this.math);
    this.graphRunner.setSession(this.session);

    this.startInference();

    this.modelInitialized = true;
  }

  private populateDatasets() {
    this.dataSets = {};
    xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON)
        .then(
            xhrDatasetConfigs => {
              for (const datasetName in xhrDatasetConfigs) {
                if (xhrDatasetConfigs.hasOwnProperty(datasetName)) {
                  this.dataSets[datasetName] =
                      new XhrDataset(xhrDatasetConfigs[datasetName]);
                }
              }
              this.datasetNames = Object.keys(this.dataSets);
              this.selectedDatasetName = this.datasetNames[0];
              this.xhrDatasetConfigs = xhrDatasetConfigs;
              this.updateSelectedDataset(this.datasetNames[0]);
            },
            error => {
              throw new Error('Dataset config could not be loaded: ' + error);
            });
  }

  private updateSelectedDataset(datasetName: string) {
    if (this.dataSet != null) {
      this.dataSet.removeNormalization(IMAGE_DATA_INDEX);
    }

    this.graphRunner.stopTraining();
    this.graphRunner.stopInferring();

    if (this.dataSet != null) {
      this.dataSet.dispose();
    }

    this.selectedDatasetName = datasetName;
    this.selectedModelName = '';
    this.dataSet = this.dataSets[datasetName];
    this.datasetDownloaded = false;
    this.showDatasetStats = false;

    this.dataSet.fetchData().then(() => {
      this.datasetDownloaded = true;
      this.applyNormalization(this.selectedNormalizationOption);
      this.setupDatasetStats();
      if (this.isValid) {
        this.createModel();
      }
      // Get prebuilt models.
      this.populateModelDropdown();
    });

    this.inputShape = this.dataSet.getDataShape(IMAGE_DATA_INDEX);
    //this.labelShape = this.dataSet.getDataShape(LABEL_DATA_INDEX);
    this.labelShape = [2];

    this.layersContainer =
        this.querySelector('#hidden-layers') as HTMLDivElement;
    this.genLayersContainer = 
        this.querySelector('#gen-hidden-layers') as HTMLDivElement;

    this.inputLayer = this.querySelector('#input-layer') as ModelLayer;
    this.inputLayer.outputShapeDisplay =
        model_builder_util.getDisplayShape(this.inputShape);

    this.genInputLayer = this.querySelector('#gen-input-layer') as ModelLayer;
    this.genInputLayer.outputShapeDisplay =
        model_builder_util.getDisplayShape(this.randVectorShape);

    const labelShapeDisplay =
        model_builder_util.getDisplayShape(this.labelShape);
    const costLayer = this.querySelector('#cost-layer') as ModelLayer;
    costLayer.inputShapeDisplay = labelShapeDisplay;
    costLayer.outputShapeDisplay = labelShapeDisplay;
    const genCostLayer = this.querySelector('#gen-cost-layer') as ModelLayer;
    genCostLayer.inputShapeDisplay = 
        model_builder_util.getDisplayShape(this.inputShape);
    genCostLayer.outputShapeDisplay = 
        model_builder_util.getDisplayShape(this.inputShape);

    const outputLayer = this.querySelector('#output-layer') as ModelLayer;
    outputLayer.inputShapeDisplay = labelShapeDisplay;
    const genOutputLayer = this.querySelector('#gen-output-layer') as ModelLayer;
    genOutputLayer.inputShapeDisplay = 
        model_builder_util.getDisplayShape(this.inputShape);

    this.buildRealImageContainer();
    this.buildFakeImageContainer();
  }

  /* Helper function for building out container for images*/
  private buildRealImageContainer() {
    const inferenceContainer =
        this.querySelector('#real-container') as HTMLElement;
    inferenceContainer.innerHTML = '';
    this.inputNDArrayVisualizers = [];
    this.outputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
      const inferenceExampleElement = document.createElement('div');
      inferenceExampleElement.className = 'inference-example';

      // Set up the input visualizer.
      const ndarrayImageVisualizer =
          document.createElement('ndarray-image-visualizer') as
          NDArrayImageVisualizer;
      ndarrayImageVisualizer.setShape(this.inputShape);
      ndarrayImageVisualizer.setSize(
          INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
      this.inputNDArrayVisualizers.push(ndarrayImageVisualizer);
      inferenceExampleElement.appendChild(ndarrayImageVisualizer);

      // Set up the output ndarray visualizer.
      const ndarrayLogitsVisualizer =
          document.createElement('ndarray-logits-visualizer') as
          NDArrayLogitsVisualizer;
      ndarrayLogitsVisualizer.initialize(
          INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
      this.outputNDArrayVisualizers.push(ndarrayLogitsVisualizer);
      inferenceExampleElement.appendChild(ndarrayLogitsVisualizer);

      inferenceContainer.appendChild(inferenceExampleElement);
    }
  }

  private buildFakeImageContainer() {
    const inferenceContainer =
        this.querySelector('#generated-container') as HTMLElement;
    inferenceContainer.innerHTML = '';
    this.fakeInputNDArrayVisualizers = [];
    this.fakeOutputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
      const inferenceExampleElement = document.createElement('div');
      inferenceExampleElement.className = 'inference-example';

      // Set up the input visualizer.
      const ndarrayImageVisualizer =
          document.createElement('ndarray-image-visualizer') as
          NDArrayImageVisualizer;
      ndarrayImageVisualizer.setShape(this.inputShape);
      ndarrayImageVisualizer.setSize(
          INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
      this.fakeInputNDArrayVisualizers.push(ndarrayImageVisualizer);
      inferenceExampleElement.appendChild(ndarrayImageVisualizer);

      // Set up the output ndarray visualizer.
      const ndarrayLogitsVisualizer =
          document.createElement('ndarray-logits-visualizer') as
          NDArrayLogitsVisualizer;
      ndarrayLogitsVisualizer.initialize(
          INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
      this.fakeOutputNDArrayVisualizers.push(ndarrayLogitsVisualizer);
      inferenceExampleElement.appendChild(ndarrayLogitsVisualizer);

      inferenceContainer.appendChild(inferenceExampleElement);
    }
  }

  private populateModelDropdown() {
    const modelNames = ['Custom'];
    const genModelNames = ['Custom'];

    const modelConfigs =
        this.xhrDatasetConfigs[this.selectedDatasetName].modelConfigs;
    for (const modelName in modelConfigs) {
      if (modelConfigs.hasOwnProperty(modelName)) {
        if (modelName.endsWith('(disc)')) {
          modelNames.push(modelName);
        } else {
          genModelNames.push(modelName);
        }
      }
    }

    this.modelNames = modelNames;
    this.genModelNames = genModelNames;
    this.selectedModelName = modelNames[modelNames.length - 1];
    this.genSelectedModelName = genModelNames[genModelNames.length - 1];
    this.updateSelectedModel(this.selectedModelName, 'disc');
    this.updateSelectedModel(this.genSelectedModelName, 'gen');
  }

  private updateSelectedModel(modelName: string, which: string) {
    this.removeAllLayers();
    if (modelName === 'Custom') {
      // TODO(nsthorat): Remember the custom layers.
      return;
    }

    this.loadModelFromPath(this.xhrDatasetConfigs[this.selectedDatasetName]
                               .modelConfigs[modelName]
                               .path,
                           which);
  }

  private loadModelFromPath(modelPath: string, which: string) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', modelPath);

    xhr.onload = () => {
      this.loadModelFromJson(xhr.responseText, which);
    };
    xhr.onerror = (error) => {
      throw new Error(
          'Model could not be fetched from ' + modelPath + ': ' + error);
    };
    xhr.send();
  }

  private setupDatasetStats() {
    this.datasetStats = this.dataSet.getStats();
    this.statsExampleCount = this.datasetStats[IMAGE_DATA_INDEX].exampleCount;
    this.statsInputRange = '[' + this.datasetStats[IMAGE_DATA_INDEX].inputMin +
        ', ' + this.datasetStats[IMAGE_DATA_INDEX].inputMax + ']';
    this.statsInputShapeDisplay = model_builder_util.getDisplayShape(
        this.datasetStats[IMAGE_DATA_INDEX].shape);
    this.statsLabelShapeDisplay = model_builder_util.getDisplayShape(
        this.datasetStats[LABEL_DATA_INDEX].shape);
    this.showDatasetStats = true;
  }

  private applyNormalization(selectedNormalizationOption: number) {
    switch (selectedNormalizationOption) {
      case Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE: {
        this.dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);
        break;
      }
      case Normalization.NORMALIZATION_ZERO_TO_ONE: {
        this.dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, 0, 1);
        break;
      }
      case Normalization.NORMALIZATION_NONE: {
        this.dataSet.removeNormalization(IMAGE_DATA_INDEX);
        break;
      }
      default: { throw new Error('Normalization option must be 0, 1, or 2'); }
    }
    this.setupDatasetStats();
  }

  private recreateCharts() {
    this.costChartData = [];
    if (this.costChart != null) {
      this.costChart.destroy();
    }
    this.costChart =
        this.createChart('cost-chart', 'Discriminator Cost', this.costChartData, 0);

    if (this.accuracyChart != null) {
      this.accuracyChart.destroy();
    }
    this.accuracyChartData = [];
    this.accuracyChart = this.createChart(
        'accuracy-chart', 'Generator Cost', this.accuracyChartData, 0);

    if (this.examplesPerSecChart != null) {
      this.examplesPerSecChart.destroy();
    }
    this.examplesPerSecChartData = [];
    this.examplesPerSecChart = this.createChart(
        'examplespersec-chart', 'Examples/sec', this.examplesPerSecChartData,
        0);
  }

  private createChart(
      canvasId: string, label: string, data: ChartData[], min?: number,
      max?: number): Chart {
    const context = (document.getElementById(canvasId) as HTMLCanvasElement)
                        .getContext('2d') as CanvasRenderingContext2D;
    return new Chart(context, {
      type: 'line',
      data: {
        datasets: [{
          data,
          fill: false,
          label,
          pointRadius: 0,
          borderColor: 'rgba(75,192,192,1)',
          borderWidth: 1,
          lineTension: 0,
          pointHitRadius: 8
        }]
      },
      options: {
        animation: {duration: 0},
        responsive: false,
        scales: {
          xAxes: [{type: 'linear', position: 'bottom'}],
          yAxes: [{
            ticks: {
              max,
              min,
            }
          }]
        }
      }
    });
  }

  displayBatchesTrained(totalBatchesTrained: number) {
    this.examplesTrained = this.batchSize * totalBatchesTrained;
  }

  displayCost(cost: Scalar, which: String) {
    if (which === 'disc') {
      this.costChartData.push(
          {x: this.graphRunner.getTotalBatchesTrained(), y: cost.get()});
      this.costChart.update();
    }

    else {
      this.accuracyChartData.push(
          {x: this.graphRunner.getTotalBatchesTrained(), y: cost.get()});
      this.accuracyChart.update();
    }
  }

  displayAccuracy(accuracy: Scalar) {
    this.accuracyChartData.push({
      x: this.graphRunner.getTotalBatchesTrained(),
      y: accuracy.get() * 100
    });
    this.accuracyChart.update();
  }

  displayInferenceExamplesPerSec(examplesPerSec: number) {
    this.inferencesPerSec =
        this.smoothExamplesPerSec(this.inferencesPerSec, examplesPerSec);
    this.inferenceDuration = Number((1000 / examplesPerSec).toPrecision(3));
  }

  displayExamplesPerSec(examplesPerSec: number) {
    this.examplesPerSecChartData.push(
        {x: this.graphRunner.getTotalBatchesTrained(), y: examplesPerSec});
    this.examplesPerSecChart.update();
    this.examplesPerSec =
        this.smoothExamplesPerSec(this.examplesPerSec, examplesPerSec);
  }

  private smoothExamplesPerSec(
      lastExamplesPerSec: number, nextExamplesPerSec: number): number {
    return Number((EXAMPLE_SEC_STAT_SMOOTHING_FACTOR * lastExamplesPerSec +
                   (1 - EXAMPLE_SEC_STAT_SMOOTHING_FACTOR) * nextExamplesPerSec)
                      .toPrecision(3));
  }

  displayInferenceExamplesOutput(
      inputFeeds: FeedEntry[][], inferenceOutputs: NDArray[][]) {

    let realImages: Array3D[] = [];
    const realLabels: Array1D[] = [];
    const realLogits: Array1D[] = [];

    let fakeImages: Array3D[] = []
    const fakeLabels: Array1D[] = [];
    const fakeLogits: Array1D[] = [];

    for (let i = 0; i < inputFeeds.length; i++) {
      realImages.push(inputFeeds[i][0].data as Array3D);
      realLabels.push(inputFeeds[i][2].data as Array1D);
      realLogits.push(inferenceOutputs[2][i] as Array1D);
      fakeImages.push((inferenceOutputs[0][i] as Array3D));
      fakeLabels.push(inputFeeds[i][3].data as Array1D);
      fakeLogits.push(inferenceOutputs[1][i] as Array1D);
    }

    realImages =
        this.dataSet.unnormalizeExamples(realImages, IMAGE_DATA_INDEX) as Array3D[];

    fakeImages = 
        this.dataSet.unnormalizeExamples(fakeImages, IMAGE_DATA_INDEX) as Array3D[];

    // Draw the images.
    for (let i = 0; i < inputFeeds.length; i++) {
      this.inputNDArrayVisualizers[i].saveImageDataFromNDArray(realImages[i]);
      this.fakeInputNDArrayVisualizers[i].saveImageDataFromNDArray(fakeImages[i]);
    }

    // Draw the logits.
    for (let i = 0; i < inputFeeds.length; i++) {
      const realSoftmaxLogits = this.math.softmax(realLogits[i]);
      const fakeSoftmaxLogits = this.math.softmax(fakeLogits[i]);

      this.outputNDArrayVisualizers[i].drawLogits(
          realSoftmaxLogits, realLabels[i],
          this.xhrDatasetConfigs[this.selectedDatasetName].labelClassNames);
      this.fakeOutputNDArrayVisualizers[i].drawLogits(
          fakeSoftmaxLogits, fakeLabels[i],
          this.xhrDatasetConfigs[this.selectedDatasetName].labelClassNames);
      this.inputNDArrayVisualizers[i].draw();
      this.fakeInputNDArrayVisualizers[i].draw();

      realSoftmaxLogits.dispose();
      fakeSoftmaxLogits.dispose();
    }
  }

  addLayer(which: string): ModelLayer {
    var layersContainer: HTMLDivElement;
    var hiddenLayers: ModelLayer[];
    var inputShape: number[];

    if (which === 'gen') {
      layersContainer = this.genLayersContainer;
      hiddenLayers = this.genHiddenLayers;
      inputShape = this.randVectorShape;
    } else {
      layersContainer = this.layersContainer;
      hiddenLayers = this.discHiddenLayers;
      inputShape = this.inputShape;
    }

    const modelLayer = document.createElement('model-layer') as ModelLayer;
    modelLayer.className = 'layer';
    layersContainer.appendChild(modelLayer);
    const lastHiddenLayer = hiddenLayers[hiddenLayers.length - 1];
    const lastOutputShape = lastHiddenLayer != null ?
        lastHiddenLayer.getOutputShape() :
        inputShape;
    hiddenLayers.push(modelLayer);
    modelLayer.initialize(this, lastOutputShape, which);
    return modelLayer;
  }

  removeLayer(modelLayer: ModelLayer, which: string) {
    if (which === 'gen') {
      this.genLayersContainer.removeChild(modelLayer);
      this.genHiddenLayers.splice(this.genHiddenLayers.indexOf(modelLayer), 1);
    } else {
      this.layersContainer.removeChild(modelLayer);
      this.discHiddenLayers.splice(this.discHiddenLayers.indexOf(modelLayer), 1);
    }
    this.layerParamChanged();
  }

  private removeAllLayers() {
    for (let i = 0; i < this.discHiddenLayers.length; i++) {
      this.layersContainer.removeChild(this.discHiddenLayers[i]);
    }
    this.discHiddenLayers = [];
    this.layerParamChanged();
  }

  private validateModel() {
    let valid = true;
    for (let i = 0; i < this.discHiddenLayers.length; ++i) {
      valid = valid && this.discHiddenLayers[i].isValid();
    }
    if (this.discHiddenLayers.length > 0) {
      const lastLayer = this.discHiddenLayers[this.discHiddenLayers.length - 1];
      valid = valid &&
          util.arraysEqual(this.labelShape, lastLayer.getOutputShape());
    }
    valid = valid && (this.discHiddenLayers.length > 0);

    for (let i = 0; i < this.genHiddenLayers.length; ++i) {
      valid = valid && this.genHiddenLayers[i].isValid();
    }
    if (this.genHiddenLayers.length > 0) {
      const lastLayer = this.genHiddenLayers[this.genHiddenLayers.length - 1];
      valid = valid &&
          util.arraysEqual(this.inputShape, lastLayer.getOutputShape());
    }
    valid = valid && (this.genHiddenLayers.length > 0);

    this.isValid = valid;
  }

  layerParamChanged() {
    // Go through each of the model layers and propagate shapes.
    let lastOutputShape = this.inputShape;
    for (let i = 0; i < this.discHiddenLayers.length; i++) {
      lastOutputShape = this.discHiddenLayers[i].setInputShape(lastOutputShape);
    }

    lastOutputShape = this.randVectorShape;
    for (let i = 0; i < this.genHiddenLayers.length; i++) {
      lastOutputShape = this.genHiddenLayers[i].setInputShape(lastOutputShape);
    }

    this.validateModel();

    if (this.isValid) {
      this.createModel();
    }
  }

  private downloadModel() {
    const modelJson = this.getModelAsJson();
    const blob = new Blob([modelJson], {type: 'text/json'});
    const textFile = window.URL.createObjectURL(blob);

    // Force a download.
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style.display = 'none';
    a.href = textFile;
    // tslint:disable-next-line:no-any
    (a as any).download = this.selectedDatasetName + '_model';
    a.click();

    document.body.removeChild(a);
    window.URL.revokeObjectURL(textFile);
  }

  private uploadModel() {
    (this.querySelector('#model-file') as HTMLInputElement).click();
  }

  private setupUploadModelButton() {
    // Show and setup the load view button.
    const fileInput = this.querySelector('#model-file') as HTMLInputElement;
    fileInput.addEventListener('change', event => {
      const file = fileInput.files[0];
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      fileInput.value = '';
      const fileReader = new FileReader();
      fileReader.onload = (evt) => {
        this.removeAllLayers();
        const modelJson: string = fileReader.result;
        this.loadModelFromJson(modelJson, 'disc');
      };
      fileReader.readAsText(file);
    });
  }

  private getModelAsJson(): string {
    const layerBuilders: LayerBuilder[] = [];
    for (let i = 0; i < this.discHiddenLayers.length; i++) {
      layerBuilders.push(this.discHiddenLayers[i].layerBuilder);
    }
    return JSON.stringify(layerBuilders);
  }

  private loadModelFromJson(modelJson: string, which: string) {
    var lastOutputShape: number[];
    var hiddenLayers: ModelLayer[];
    if (which === 'disc') {
      lastOutputShape = this.inputShape;
      hiddenLayers = this.discHiddenLayers;
    } else {
      lastOutputShape = this.randVectorShape;
      hiddenLayers = this.genHiddenLayers;
    }

    const layerBuilders = JSON.parse(modelJson) as LayerBuilder[];
    for (let i = 0; i < layerBuilders.length; i++) {
      const modelLayer = this.addLayer(which);
      modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
      lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);
    }
    this.validateModel();
  }

  private uploadWeights() {
    (this.querySelector('#weights-file') as HTMLInputElement).click();
  }

  private setupUploadWeightsButton() {
    // Show and setup the load view button.
    const fileInput = this.querySelector('#weights-file') as HTMLInputElement;
    fileInput.addEventListener('change', event => {
      const file = fileInput.files[0];
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      fileInput.value = '';
      const fileReader = new FileReader();
      fileReader.onload = (evt) => {
        const weightsJson: string = fileReader.result;
        this.loadWeightsFromJson(weightsJson);
        this.createModel();
      };
      fileReader.readAsText(file);
    });
  }

  private loadWeightsFromJson(weightsJson: string) {
    this.loadedWeights = JSON.parse(weightsJson) as LayerWeightsDict[];
  }
}

document.registerElement(GANPlayground.prototype.is, GANPlayground);
