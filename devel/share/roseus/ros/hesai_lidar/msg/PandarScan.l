;; Auto-generated. Do not edit!


(when (boundp 'hesai_lidar::PandarScan)
  (if (not (find-package "HESAI_LIDAR"))
    (make-package "HESAI_LIDAR"))
  (shadow 'PandarScan (find-package "HESAI_LIDAR")))
(unless (find-package "HESAI_LIDAR::PANDARSCAN")
  (make-package "HESAI_LIDAR::PANDARSCAN"))

(in-package "ROS")
;;//! \htmlinclude PandarScan.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass hesai_lidar::PandarScan
  :super ros::object
  :slots (_header _packets ))

(defmethod hesai_lidar::PandarScan
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:packets __packets) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _packets __packets)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:packets
   (&rest __packets)
   (if (keywordp (car __packets))
       (send* _packets __packets)
     (progn
       (if __packets (setq _packets (car __packets)))
       _packets)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; hesai_lidar/PandarPacket[] _packets
    (apply #'+ (send-all _packets :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; hesai_lidar/PandarPacket[] _packets
     (write-long (length _packets) s)
     (dolist (elem _packets)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; hesai_lidar/PandarPacket[] _packets
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _packets (let (r) (dotimes (i n) (push (instance hesai_lidar::PandarPacket :init) r)) r))
     (dolist (elem- _packets)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get hesai_lidar::PandarScan :md5sum-) "70c3ed4f4ae9144323298b04cc2c760b")
(setf (get hesai_lidar::PandarScan :datatype-) "hesai_lidar/PandarScan")
(setf (get hesai_lidar::PandarScan :definition-)
      "Header header
PandarPacket[] packets

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: hesai_lidar/PandarPacket
# field		size(byte)
# SOB 		2
# angle		2
# measure	5
# block		SOB + angle + measure * 40
# timestamp	4
# factory	2
# reserve	8
# rpm		2
# tail		timestamp + factory + reserve + rpm
# packet	block * 6 + tail

time stamp
uint8[] data
uint32 size
")



(provide :hesai_lidar/PandarScan "70c3ed4f4ae9144323298b04cc2c760b")


