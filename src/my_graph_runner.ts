import {Array3D, InputProvider, Tensor, Optimizer, CostReduction, FeedEntry, Session,
NDArrayMath, NDArray, Scalar, GraphRunnerEventObserver} from 'deeplearn';

const DEFAULT_EVAL_INTERVAL_MS = 1500;
const DEFAULT_COST_INTERVAL_MS = 500;
const DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS = 3000;


export interface MyGraphRunnerEventObserver {
  batchesTrainedCallback?: (totalBatchesTrained: number) => void;
  avgCostCallback?: (avgCost: Scalar) => void;
  metricCallback?: (metric: NDArray) => void;
  inferenceExamplesCallback?:
      (feeds: FeedEntry[][], inferenceValues: NDArray[][]) => void;
  inferenceExamplesPerSecCallback?: (examplesPerSec: number) => void;
  trainExamplesPerSecCallback?: (examplesPerSec: number) => void;
  totalTimeCallback?: (totalTimeSec: number) => void;
  doneTrainingCallback?: () => void;
}

export enum MetricReduction {
  SUM,
  MEAN
}

/**
 * A class that drives the training of a graph model given a dataset. It allows
 * the user to provide a set of callbacks for measurements like cost, accuracy,
 * and speed of training.
 */
export class MyGraphRunner {
  private costTensor: Tensor;
  private trainFeedEntries: FeedEntry[];
  private batchSize: number;
  private optimizer: Optimizer;
  private currentTrainLoopNumBatches: number|undefined;
  private costIntervalMs: number;

  private metricTensor: Tensor|undefined;
  private metricFeedEntries: FeedEntry[]|undefined;
  private metricBatchSize: number|undefined;
  private metricReduction: MetricReduction;
  private metricIntervalMs: number;

  private genImageTensor: Tensor;
  private discPredictionFakeTensor: Tensor;
  private discPredictionRealTensor: Tensor;
  private inferenceFeedEntries: FeedEntry[]|undefined;
  private inferenceExampleIntervalMs: number;
  private inferenceExampleCount: number;

  // Runtime information.
  private isTraining: boolean;
  private totalBatchesTrained: number;
  private batchesTrainedThisRun: number;
  private lastComputedMetric: NDArray;

  private isInferring: boolean;
  private lastInferTimeoutID: number;
  private currentInferenceLoopNumPasses: number|undefined;
  private inferencePassesThisRun: number;

  private trainStartTimestamp: number;
  private lastCostTimestamp = 0;
  private lastEvalTimestamp = 0;

  private lastStopTimestamp: number|null;
  private totalIdleTimeMs = 0;

  private zeroScalar: Scalar;
  private metricBatchSizeScalar: Scalar;

  constructor(
      private math: NDArrayMath, private session: Session,
      private eventObserver: MyGraphRunnerEventObserver) {
    this.resetStatistics();
    this.zeroScalar = Scalar.new(0);
  }

  resetStatistics() {
    this.totalBatchesTrained = 0;
    this.totalIdleTimeMs = 0;
    this.lastStopTimestamp = null;
  }

  /**
   * Start the training loop with an optional number of batches to train for.
   * Optionally takes a metric tensor and feed entries to compute periodically.
   * This can be used for computing accuracy, or a similar metric.
   */
  train(
      costTensor: Tensor, trainFeedEntries: FeedEntry[], batchSize: number,
      optimizer: Optimizer, numBatches?: number, metricTensor?: Tensor,
      metricFeedEntries?: FeedEntry[], metricBatchSize?: number,
      metricReduction = MetricReduction.MEAN,
      evalIntervalMs = DEFAULT_EVAL_INTERVAL_MS,
      costIntervalMs = DEFAULT_COST_INTERVAL_MS) {
    this.costTensor = costTensor;
    this.trainFeedEntries = trainFeedEntries;
    this.metricTensor = metricTensor;
    this.metricFeedEntries = metricFeedEntries;
    if (metricBatchSize != null && this.metricBatchSize !== metricBatchSize) {
      if (this.metricBatchSizeScalar != null) {
        this.metricBatchSizeScalar.dispose();
      }
      this.metricBatchSizeScalar = Scalar.new(metricBatchSize);
    }
    this.metricBatchSize = metricBatchSize;
    this.metricReduction = metricReduction;
    this.batchSize = batchSize;
    this.optimizer = optimizer;

    this.metricIntervalMs = evalIntervalMs;
    this.costIntervalMs = costIntervalMs;
    this.currentTrainLoopNumBatches = numBatches;

    this.batchesTrainedThisRun = 0;
    this.isTraining = true;
    this.trainStartTimestamp = performance.now();
    this.trainNetwork();
  }

  stopTraining() {
    this.isTraining = false;
    this.lastStopTimestamp = performance.now();
  }

  resumeTraining() {
    this.isTraining = true;
    if (this.lastStopTimestamp != null) {
      this.totalIdleTimeMs += performance.now() - this.lastStopTimestamp;
    }
    this.trainNetwork();
  }

  private trainNetwork() {
    if (this.batchesTrainedThisRun === this.currentTrainLoopNumBatches) {
      this.stopTraining();
    }

    if (!this.isTraining) {
      if (this.eventObserver.doneTrainingCallback != null) {
        this.eventObserver.doneTrainingCallback();
      }
      return;
    }

    const start = performance.now();
    const shouldComputeCost = this.eventObserver.avgCostCallback != null &&
        (start - this.lastCostTimestamp > this.costIntervalMs);
    if (shouldComputeCost) {
      this.lastCostTimestamp = start;
    }

    const costReduction =
        shouldComputeCost ? CostReduction.MEAN : CostReduction.NONE;

    this.math.scope((keep) => {
      const avgCost = this.session.train(
          this.costTensor, this.trainFeedEntries, this.batchSize,
          this.optimizer, costReduction);

      if (shouldComputeCost) {
        const trainTime = performance.now() - start;

        this.eventObserver.avgCostCallback(avgCost);

        if (this.eventObserver.trainExamplesPerSecCallback != null) {
          const examplesPerSec = (this.batchSize * 1000 / trainTime);
          this.eventObserver.trainExamplesPerSecCallback(examplesPerSec);
        }
      }

      if (this.eventObserver.metricCallback != null &&
          this.metricFeedEntries != null &&
          start - this.lastEvalTimestamp > this.metricIntervalMs) {
        this.lastEvalTimestamp = start;

        if (this.lastComputedMetric != null) {
          this.lastComputedMetric.dispose();
        }
        this.lastComputedMetric = this.computeMetric();
        this.eventObserver.metricCallback(this.lastComputedMetric);
      }

      if (this.eventObserver.totalTimeCallback != null) {
        this.eventObserver.totalTimeCallback(
            (start - this.trainStartTimestamp) / 1000);
      }

      this.batchesTrainedThisRun++;
      this.totalBatchesTrained++;

      if (this.eventObserver.batchesTrainedCallback != null) {
        this.eventObserver.batchesTrainedCallback(this.totalBatchesTrained);
      }

    });
    requestAnimationFrame(() => this.trainNetwork());
  }


  infer(
      genImageTensor: Tensor, discPredictionFakeTensor: Tensor,
      discPredictionRealTensor: Tensor, inferenceFeedEntries: FeedEntry[],
      inferenceExampleIntervalMs = DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS,
      inferenceExampleCount = 5, numPasses?: number) {
    if (this.eventObserver.inferenceExamplesCallback == null &&
        this.eventObserver.inferenceExamplesPerSecCallback == null) {
      throw new Error(
          'Cannot start inference loop, no inference example or ' +
          'examples/sec observer provided.');
    }

    // Make sure the feed values are providers, and not NDArrays.
    for (let i = 0; i < inferenceFeedEntries.length; i++) {
      const feedEntry = inferenceFeedEntries[i];

      if (feedEntry.data instanceof NDArray) {
        throw new Error(
            'Cannot start inference on the model runner with feed entries of ' +
            'type NDArray. Please use InputProviders.');
      }
    }

    this.inferenceExampleIntervalMs = inferenceExampleIntervalMs;
    this.genImageTensor = genImageTensor;
    this.discPredictionFakeTensor = discPredictionFakeTensor;
    this.discPredictionRealTensor = discPredictionRealTensor;
    this.inferenceFeedEntries = inferenceFeedEntries;
    this.inferenceExampleCount = inferenceExampleCount;
    this.currentInferenceLoopNumPasses = numPasses;
    if (!this.isInferring) {
      this.inferencePassesThisRun = 0;
      requestAnimationFrame(() => this.inferNetwork());
    }
    this.isInferring = true;
  }

  private inferNetwork() {
    if (!this.isInferring ||
        this.inferencePassesThisRun === this.currentInferenceLoopNumPasses) {
      return;
    }

    this.math.scope((keep, track) => {
      const feeds: FeedEntry[][] = [];
      const genImageValues: NDArray[] = [];
      const discPredictionFakeValues: NDArray[] = [];
      const discPredictionRealValues: NDArray[] = [];

      const start = performance.now();
      for (let i = 0; i < this.inferenceExampleCount; i++) {
        // Populate a new FeedEntry[] populated with NDArrays.
        const ndarrayFeedEntries: FeedEntry[] = [];
        for (let j = 0; j < this.inferenceFeedEntries.length; j++) {
          const feedEntry = this.inferenceFeedEntries[j];
          const nextData = track((feedEntry.data as InputProvider).getNextCopy(this.math));
          const moreData = track((feedEntry.data as InputProvider).getNextCopy(this.math));

          console.log(nextData);
          console.log(moreData);
          console.log(track((feedEntry.data as InputProvider).getNextCopy(this.math)));

          //const toPush = {
          //  tensor: feedEntry.tensor,
          //  data: moreData
          //};
          //ndarrayFeedEntries.push(toPush);
        }
        feeds.push(ndarrayFeedEntries);
        console.log(ndarrayFeedEntries);

        const evaluatedTensors = this.session.evalAll(
          [this.genImageTensor, this.discPredictionFakeTensor, this.discPredictionRealTensor],
          ndarrayFeedEntries
        );

        genImageValues.push(evaluatedTensors[0]);
        discPredictionFakeValues.push(evaluatedTensors[1]);
        discPredictionRealValues.push(evaluatedTensors[2]);
      }

      if (this.eventObserver.inferenceExamplesPerSecCallback != null) {
        // Force a GPU download, since inference results are generally needed on
        // the CPU and it's more fair to include blocking on the GPU to complete
        // its work for the inference measurement.

        const inferenceExamplesPerSecTime = performance.now() - start;

        const examplesPerSec =
            (this.inferenceExampleCount * 1000 / inferenceExamplesPerSecTime);
        this.eventObserver.inferenceExamplesPerSecCallback(examplesPerSec);
      }

      console.log(feeds[0][0].data as Array3D);

      if (this.eventObserver.inferenceExamplesCallback != null) {
        this.eventObserver.inferenceExamplesCallback(
          feeds,
          [genImageValues, discPredictionFakeValues, discPredictionRealValues]
        );
      }
      this.inferencePassesThisRun++;

    });
    this.lastInferTimeoutID = window.setTimeout(
        () => this.inferNetwork(), this.inferenceExampleIntervalMs);
  }

  stopInferring() {
    this.isInferring = false;
    window.clearTimeout(this.lastInferTimeoutID);
  }

  isInferenceRunning(): boolean {
    return this.isInferring;
  }

  computeMetric(): Scalar {
    if (this.metricFeedEntries == null) {
      throw new Error('Cannot compute metric, no metric FeedEntries provided.');
    }

    let metric = this.zeroScalar;

    return this.math.scope((keep) => {
      for (let i = 0; i < this.metricBatchSize; i++) {
        const metricValue =
            this.session.eval(this.metricTensor, this.metricFeedEntries);

        metric = this.math.add(metric, metricValue);
      }

      if (this.metricReduction === MetricReduction.MEAN) {
        metric = this.math.divide(metric, this.metricBatchSizeScalar);
      }

      return metric;
    });
  }

  getTotalBatchesTrained(): number {
    return this.totalBatchesTrained;
  }

  getLastComputedMetric(): Scalar {
    return this.lastComputedMetric;
  }

  setMath(math: NDArrayMath) {
    this.math = math;
  }

  setSession(session: Session) {
    this.session = session;
  }

  setInferenceExampleCount(inferenceExampleCount: number) {
    this.inferenceExampleCount = inferenceExampleCount;
  }
}
