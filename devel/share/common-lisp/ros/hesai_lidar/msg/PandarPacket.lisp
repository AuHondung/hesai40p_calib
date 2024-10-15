; Auto-generated. Do not edit!


(cl:in-package hesai_lidar-msg)


;//! \htmlinclude PandarPacket.msg.html

(cl:defclass <PandarPacket> (roslisp-msg-protocol:ros-message)
  ((stamp
    :reader stamp
    :initarg :stamp
    :type cl:real
    :initform 0)
   (data
    :reader data
    :initarg :data
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0))
   (size
    :reader size
    :initarg :size
    :type cl:integer
    :initform 0))
)

(cl:defclass PandarPacket (<PandarPacket>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PandarPacket>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PandarPacket)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name hesai_lidar-msg:<PandarPacket> is deprecated: use hesai_lidar-msg:PandarPacket instead.")))

(cl:ensure-generic-function 'stamp-val :lambda-list '(m))
(cl:defmethod stamp-val ((m <PandarPacket>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hesai_lidar-msg:stamp-val is deprecated.  Use hesai_lidar-msg:stamp instead.")
  (stamp m))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <PandarPacket>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hesai_lidar-msg:data-val is deprecated.  Use hesai_lidar-msg:data instead.")
  (data m))

(cl:ensure-generic-function 'size-val :lambda-list '(m))
(cl:defmethod size-val ((m <PandarPacket>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hesai_lidar-msg:size-val is deprecated.  Use hesai_lidar-msg:size instead.")
  (size m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PandarPacket>) ostream)
  "Serializes a message object of type '<PandarPacket>"
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'stamp)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'stamp) (cl:floor (cl:slot-value msg 'stamp)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'data))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'size)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'size)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'size)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'size)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PandarPacket>) istream)
  "Deserializes a message object of type '<PandarPacket>"
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'stamp) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'data)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream)))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'size)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'size)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'size)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'size)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PandarPacket>)))
  "Returns string type for a message object of type '<PandarPacket>"
  "hesai_lidar/PandarPacket")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PandarPacket)))
  "Returns string type for a message object of type 'PandarPacket"
  "hesai_lidar/PandarPacket")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PandarPacket>)))
  "Returns md5sum for a message object of type '<PandarPacket>"
  "c29f0f7365a75504f5f0008b5913cb94")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PandarPacket)))
  "Returns md5sum for a message object of type 'PandarPacket"
  "c29f0f7365a75504f5f0008b5913cb94")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PandarPacket>)))
  "Returns full string definition for message of type '<PandarPacket>"
  (cl:format cl:nil "# field		size(byte)~%# SOB 		2~%# angle		2~%# measure	5~%# block		SOB + angle + measure * 40~%# timestamp	4~%# factory	2~%# reserve	8~%# rpm		2~%# tail		timestamp + factory + reserve + rpm~%# packet	block * 6 + tail~%~%time stamp~%uint8[] data~%uint32 size~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PandarPacket)))
  "Returns full string definition for message of type 'PandarPacket"
  (cl:format cl:nil "# field		size(byte)~%# SOB 		2~%# angle		2~%# measure	5~%# block		SOB + angle + measure * 40~%# timestamp	4~%# factory	2~%# reserve	8~%# rpm		2~%# tail		timestamp + factory + reserve + rpm~%# packet	block * 6 + tail~%~%time stamp~%uint8[] data~%uint32 size~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PandarPacket>))
  (cl:+ 0
     8
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PandarPacket>))
  "Converts a ROS message object to a list"
  (cl:list 'PandarPacket
    (cl:cons ':stamp (stamp msg))
    (cl:cons ':data (data msg))
    (cl:cons ':size (size msg))
))
