import {NDArrayMath, NDArray, InputProvider} from 'deeplearn';


export function getRandomInputProvider(shape: number[]): InputProvider {
  return {
    getNextCopy(math: NDArrayMath): NDArray {
      return NDArray.randNormal(shape);
    },
    disposeCopy(math: NDArrayMath, copy: NDArray) {
      copy.dispose();
    }
  }
}
