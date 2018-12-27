//
//  CLMConfig.h
//
//  Created by RongYuan on 8/3/18.
//  Copyright © 2018 vipshop. All rights reserved.
//

#ifndef CLMConfig_h
#define CLMConfig_h


#define TRACK_HAND false
#define USE_CLM true /* Switch off CLM here [Default on] */

#define PCA_DIM 23 /* Recommended 23 to 52 [Lower dimension gives higher speed] */

/* -----DO NOT MODIFY----- */
#define NO_PREV_TRACK -1
#define REDETECTION 0
#define OPTICAL_FLOW 1
#define TEMPLATE_TRACKING 2

#define TRACKING_DOWNSAMPLE_RATE 4
#define MAX_TRACKING_DIST 5
#define MIN_LEVEL_PATCH_EXPERT 0
#define EVERY_TRACKING_REINIT 3
/* ---------END---------- */

#endif /* CLMConfig_h */
