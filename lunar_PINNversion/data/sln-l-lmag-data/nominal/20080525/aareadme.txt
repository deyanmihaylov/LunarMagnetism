PDS_VERSION_ID       = PDS3 

RECORD_TYPE          = STREAM
SPACECRAFT_NAME      = SELENE
TARGET_NAME          = MOON
OBJECT               = TEXT
  PUBLICATION_DATE   = 
  NOTE               = "SELENE Level-2 Products"
END_OBJECT           = TEXT
END



                         SELENE LEVEL-2 PRODUCTS


  Introduction
  ------------

  SELENE Level-2 (L2) Products contain all of processed (calibrated) data of
  each instruments. This volume includes all data products that you ordered
  by "SOAC search and ordering service" - www.soac.selene.isas.jaxa.jp.


  File Formats
  ------------

   Contents of L2 Products
   L2 products; xxxxxxxx.sl2 is a tar file that include;
   - Data product
     data files and label files or labeled data files.
   - Preview Image
     Some L2 products have a preview image as a JPEG file.(xxxxxxx.jpg)
   - Catalog file
     xxxxxxxx.ctg is a catalog data for archive system.


  Volume Contents
  ---------------

  L2 products includes the following files and directories.

  Root directory
    AAREADME.TXT - the file you are reading
    VOLDESC.CAT  - volume description for the PDS catalog

    INDEX directory
      INDXINFO.TXT - description of contents of this directory
      INDEX.TAB    - index of all products in the archive
      INDEX.LBL    - PDS label that describes INDEX.TAB

    CATALOG directory
      CATINFO.TXT  - description of contents of this directory
      *.CAT        - descriptions of dataset, instrument, spacecraft,
                     mission, references, and personnel

    SELENE directory
      *.SL2        - data files


  Contacts
  --------

  For questions concerning these data products, contact:

    SOAC SELENE Operation and data Analysis Center:
    Z-SELENE_DB@jaxa.jp
    http://l2db.selene.darts.isas.jaxa.jp/

    SELENE project:
    http://www.selene.jaxa.jp
