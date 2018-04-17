package com.javferna.packtpub.mastering.knn.parallel.group;

import com.javferna.packtpub.mastering.knn.data.Distance;
import com.javferna.packtpub.mastering.knn.data.Sample;
import com.javferna.packtpub.mastering.knn.distances.EuclideanDistanceCalculator;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Coarse-grained concurrent version of the Knn algorithm
 * @author author
 *
 */
public class KnnClassifierParallelNew {

	/**
	 * Train data
	 */
	private List<? extends Sample> dataSet;

	/**
	 * K parameter
	 */
	private int k;

	/**
	 * Executor to execute the concurrent tasks
	 */
	private ThreadPoolExecutor executor;

	/**
	 * Number of threads to configure the executor
	 */
	private int numThreads;

	/**
	 * Check to indicate if we use the serial or the parallel sorting
	 */
	private boolean parallelSort;

	/**
	 * Constructor of the class. Initialize the internal data
	 * @param dataSet Train data set
	 * @param k K parameter
	 * @param factor Factor of increment of the number of cores
	 * @param parallelSort Check to indicate if we use the serial or the parallel sorting
	 */
	public KnnClassifierParallelNew(List<? extends Sample> dataSet, int k, int factor, boolean parallelSort) {
		this.dataSet=dataSet;
		this.k=k;
		numThreads=factor*(Runtime.getRuntime().availableProcessors());
		executor=(ThreadPoolExecutor) Executors.newFixedThreadPool(numThreads);
		this.parallelSort=parallelSort;
	}
	
	/**
	 * Method that classify an example
	 * @return Class or tag of the example
	 * @throws Exception Exception if something goes wrong
	 */
	public String[] classify (List<? extends Sample> testSet) throws Exception {

		String[] tags = new String[testSet.size()];
		
//		Distance[][] distances=new Distance[testSet.size()][dataSet.size()];
		CountDownLatch countDownLatch =new CountDownLatch(testSet.size());

		for(int i=0; i<testSet.size(); i++) {
			/*  This is the old way to create a runnable instance */
//			GroupDistanceTaskNew groupDistanceTaskNew = new GroupDistanceTaskNew(0, dataSet.size(), dataSet,
//					testSet.get(i), countDownLatch, tags, parallelSort, i, k);
//			executor.execute(groupDistanceTaskNew);

			/* This is the new way to create a runnable using lambda */
			Distance[] distances = new Distance[dataSet.size()];
			int start = 0;
			int end = dataSet.size();
			Sample sample = testSet.get(i);
			int currentIndex = i;

			executor.execute(() -> {
				for (int index = start; index < end; index++) {
					Sample localExample=dataSet.get(index);
					distances[index] = new Distance();
					distances[index].setIndex(index);
					distances[index].setDistance(EuclideanDistanceCalculator
							.calculate(localExample, sample));
				}

				if (parallelSort) {
					Arrays.parallelSort(distances);
				} else {
					Arrays.sort(distances);
				}

				Map<String, Integer> results = new HashMap<>();
				for (int m = 0; m < k; m++) {
					Sample localExample = dataSet.get(distances[m].getIndex());
					String tag = localExample.getTag();
					results.merge(tag, 1, (a, b) -> a+b);
				}

				tags[currentIndex] = Collections.max(results.entrySet(), Map.Entry.comparingByValue()).getKey();
			    countDownLatch.countDown();

			});
		}

		countDownLatch.await();

		return tags;
	}
	
	/**
	 * Method that finish the execution of the executor
	 */
	public void destroy() {
		executor.shutdown();
		try {
			executor.awaitTermination(100, TimeUnit.SECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
}
