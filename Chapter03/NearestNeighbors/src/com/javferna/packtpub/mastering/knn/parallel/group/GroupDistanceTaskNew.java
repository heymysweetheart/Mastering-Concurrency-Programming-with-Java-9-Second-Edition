package com.javferna.packtpub.mastering.knn.parallel.group;

import com.javferna.packtpub.mastering.knn.data.Distance;
import com.javferna.packtpub.mastering.knn.data.Sample;
import com.javferna.packtpub.mastering.knn.distances.EuclideanDistanceCalculator;

import java.util.*;
import java.util.concurrent.CountDownLatch;

/**
 * Task that of the coarse-grained concurrent version 
 * @author author
 *
 */
public class GroupDistanceTaskNew implements Runnable {

	/**
	 * Array of distances
	 */
	private final Distance[] distances;

	/**
	 * Indexes that determines the examples of the train data this task will process
	 */
	private final int startIndex, endIndex;

	/**
	 * Example of the test data we want to classify
	 */
	private final Sample example;

	/**
	 * Data set with the train data examples
	 */
	private final List<? extends Sample> dataSet;

	private final CountDownLatch countDownLatch;

	private final String[] tags;

	private final int index;//index tag in the target tags

	private final int k;// k of knn

	private final boolean parallelSort;

	/**
	 * Constructor of the class. Initializes all the internal data
	 * @param startIndex Start index that determines the examples of the train data this task will process
	 * @param endIndex End index that determines the examples of the train data this task will process
	 * @param dataSet Data set with the train data examples
	 * @param example Example of the test data we want to classify
	 */
	public GroupDistanceTaskNew(int startIndex,
                                int endIndex, List<? extends Sample> dataSet, Sample example, CountDownLatch countDownLatch, String[] tags, boolean parallelSort, int i, int k) {
		this.distances = new Distance[dataSet.size()];
		this.startIndex = startIndex;
		this.endIndex = endIndex;
		this.example = example;
		this.dataSet = dataSet;
		this.countDownLatch = countDownLatch;
		this.tags = tags;
		this.parallelSort = parallelSort;
		this.index = i;
		this.k = k;
	}

	@Override
	/**
	 * Concurrent task that calculates the distance between the example and the train instances between
	 * the startIndex and the endIndex 
	 */
	public void run() {
		for (int index = startIndex; index < endIndex; index++) {
			Sample localExample=dataSet.get(index);
			distances[index] = new Distance();
			distances[index].setIndex(index);
				distances[index].setDistance(EuclideanDistanceCalculator
						.calculate(localExample, example));
		}



		if (parallelSort) {
			Arrays.parallelSort(distances);
		} else {
			Arrays.sort(distances);
		}

		Map<String, Integer> results = new HashMap<>();
		for (int i = 0; i < k; i++) {
			Sample localExample = dataSet.get(distances[i].getIndex());
			String tag = localExample.getTag();
			results.merge(tag, 1, (a, b) -> a+b);
		}

		tags[index] = Collections.max(results.entrySet(), Map.Entry.comparingByValue()).getKey();
//		System.out.printf("Tags[%d] is %s\n", index, tags[index]);
		countDownLatch.countDown();
	}

}
