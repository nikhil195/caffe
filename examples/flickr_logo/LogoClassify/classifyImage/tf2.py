import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    argparser.add_argument("--image_file", help="Absolute path to image file\n",
                           type=str, required=True)

    args = argparser.parse_args()
    img_path = args.image_file
    print "img_path ", img_path

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
