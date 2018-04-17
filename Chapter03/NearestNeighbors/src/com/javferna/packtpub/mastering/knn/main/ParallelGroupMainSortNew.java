package com.javferna.packtpub.mastering.knn.main;

import com.javferna.packtpub.mastering.knn.data.BankMarketing;
import com.javferna.packtpub.mastering.knn.loader.BankMarketingLoader;
import com.javferna.packtpub.mastering.knn.parallel.group.KnnClassifierParallelGroup;
import com.javferna.packtpub.mastering.knn.parallel.group.KnnClassifierParallelNew;

import java.util.Date;
import java.util.List;
import java.util.concurrent.CountDownLatch;

/**
 * Main class that launches the tests using the coarse-grained concurrent
 * version and serial sorting
 * 
 * @author author
 *
 */
public class ParallelGroupMainSortNew {

	public static void main(String[] args) {

		BankMarketingLoader loader = new BankMarketingLoader();
		List<BankMarketing> train = loader.load(DataUtil.BankPath);
		System.out.println("Train: " + train.size());
		List<BankMarketing> test = loader.load(DataUtil.TestPath);
		System.out.println("Test: " + test.size());
		int success = 0, mistakes = 0;
		int testSize = 940;

		int k = 10;
		if (args.length==1) {
			k = Integer.parseInt(args[0]);
		}
		
		success = 0;
		mistakes = 0;

		KnnClassifierParallelNew classifier = new KnnClassifierParallelNew(train, k, 1, true);

		Date start = new Date();

		List<BankMarketing> marketings = test.subList(0, testSize);

		try {
			String[] classify = classifier.classify(marketings);

			for(int i=0;i<testSize;i++) {
				if(classify[i].equals(test.get(i).getTag())) {
					success++;
				} else {
					mistakes++;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			classifier.destroy();
		}

		Date end = new Date();

		System.out.println("******************************************");
		System.out.println("Parallel Classifier Group - K: " + k + " - Factor 1 - Parallel Sort: true");
		System.out.println("Success: " + success);
		System.out.println("Mistakes: " + mistakes);
		System.out.println("Execution Time: " + ((end.getTime() - start.getTime())/ (float)1000) + " seconds.");
		System.out.println("******************************************");

	}

}
