package com.javferna.packtpub.mastering.knn.parallel.group;

import com.javferna.packtpub.mastering.knn.data.Distance;
import com.javferna.packtpub.mastering.knn.data.Sample;

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
		
		Distance[][] distances=new Distance[testSet.size()][dataSet.size()];
		CountDownLatch countDownLatch =new CountDownLatch(testSet.size());

		for(int i=0; i<testSet.size(); i++) {
			GroupDistanceTaskNew groupDistanceTaskNew = new GroupDistanceTaskNew(distances[i], 0, dataSet.size(), dataSet,
					testSet.get(i), countDownLatch, tags, parallelSort, i, k);
			executor.execute(groupDistanceTaskNew);
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
