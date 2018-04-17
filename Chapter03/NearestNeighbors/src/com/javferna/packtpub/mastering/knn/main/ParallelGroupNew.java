package com.javferna.packtpub.mastering.knn.main;

import com.javferna.packtpub.mastering.knn.data.BankMarketing;
import com.javferna.packtpub.mastering.knn.loader.BankMarketingLoader;

import java.io.File;
import java.util.List;

//Parallel by test data set
public class ParallelGroupNew {
    public static void main(String[] args) {
        System.out.println(new File("").getAbsolutePath());
        BankMarketingLoader loader = new BankMarketingLoader();
        List<BankMarketing> train = loader.load(DataUtil.BankPath);
        System.out.println("Train: " + train.size());
        List<BankMarketing> test = loader.load(DataUtil.TestPath);
        System.out.println("Test: " + test.size());
        double currentTime = 0d;
        int success = 0, mistakes = 0;

        int k = 10;
        if (args.length==1) {
            k = Integer.parseInt(args[0]);
        }
    }
}
