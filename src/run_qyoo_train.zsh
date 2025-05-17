#!/usr/bin/env zsh

while true; do
  echo "[`date`] starting (or resuming) YOLO training…" 

  yolo task=segment mode=train \
       resume=True \
       model=../runs/segment/train_full/weights/last.pt \
       data=dataset.yaml \
       imgsz=512 batch=48 epochs=120 \
       device=mps workers=0 cache=none \
       close_mosaic=0 nms=True \
       save=True save_period=2 patience=20 \
       name=train_full

  ret=$?
  if [[ $ret -eq 0 ]]; then
    echo "[`date`] training finished cleanly – exiting loop."
    break
  fi

  echo "[`date`] YOLO exited with code $ret – sleeping 60 s then restarting…"
  sleep 60
done
