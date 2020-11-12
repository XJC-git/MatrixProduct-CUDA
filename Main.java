package com.company;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws IOException {
        for(int i = 0;i<200000000;i++) {
            DataOutputStream out = new DataOutputStream(new FileOutputStream("D:/Cpppppppp/ldFeature.bin", true));
            float min = 1f;
            float max = 10f;
            float floatBounded = min + new Random().nextFloat() * (max - min);
            //System.out.println(floatBounded);
            byte[] temp = intToByte(Float.floatToIntBits(floatBounded));
            System.out.println(temp);
            out.write(temp);
            out.close();
        }
    }
    public static byte[] intToByte(int number) {
        int temp = number;
        byte[] b = new byte[4];
        for (int i = 0; i < b.length; i++) {
            b[i] = new Integer(temp & 0xff).byteValue();// 将最低位保存在最低位
            temp = temp >> 8;// 向右移8位
        }
        return b;
    }
}
