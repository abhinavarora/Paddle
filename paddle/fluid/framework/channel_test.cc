/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/channel.h"

#include <chrono>
#include <thread>
#include "gtest/gtest.h"

using paddle::framework::Channel;
using paddle::framework::ChannelHolder;
using paddle::framework::MakeChannel;
using paddle::framework::CloseChannel;

// This tests that destroying a channel unblocks
//  any senders waiting for channel to have write space
void ChannelDestroyUnblockSenders(Channel<int> *ch, bool isBuffered) {
  size_t num_threads = 5;
  std::thread t[num_threads];
  bool thread_ended[num_threads];
  bool send_success[num_threads];

  // Launches threads that try to write and are blocked because of no readers
  for (size_t i = 0; i < num_threads; i++) {
    thread_ended[i] = false;
    send_success[i] = false;
    t[i] = std::thread(
        [&](bool *ended, bool *success) {
          int data = 10;
          bool is_exception = false;
          try {
            ch->Send(&data);
          } catch (paddle::platform::EnforceNotMet e) {
            is_exception = true;
          }
          *success = !is_exception;
          *ended = true;
        },
        &thread_ended[i], &send_success[i]);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait 0.2 sec

  // Explicitly destroy the channel
  delete ch;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait

  // Join all threads
  for (size_t i = 0; i < num_threads; i++) t[i].join();
}

// This tests that destroying a channel also unblocks
//  any receivers waiting on the channel
void ChannelDestroyUnblockReceivers(Channel<int> *ch) {
  size_t num_threads = 5;
  std::thread t[num_threads];
  bool thread_ended[num_threads];

  // Launches threads that try to read and are blocked because of no writers
  for (size_t i = 0; i < num_threads; i++) {
    thread_ended[i] = false;
    t[i] = std::thread(
        [&](bool *p) {
          int data;
          // All reads should return false
          EXPECT_EQ(ch->Receive(&data), false);
          *p = true;
        },
        &thread_ended[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // wait

  // Verify that all threads are blocked
  for (size_t i = 0; i < num_threads; i++) {
    EXPECT_EQ(thread_ended[i], false);
  }
  // delete the channel
  delete ch;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // wait
  // Verify that all threads got unblocked
  for (size_t i = 0; i < num_threads; i++) {
    EXPECT_EQ(thread_ended[i], true);
  }

  for (size_t i = 0; i < num_threads; i++) t[i].join();
}

TEST(Channel, BufferedChannelDestroyUnblocksReceiversTest) {
  size_t buffer_size = 1;
  auto ch = MakeChannel<int>(buffer_size);
  ChannelDestroyUnblockReceivers(ch);
}

TEST(Channel, BufferedChannelDestroyUnblocksSendersTest) {
  size_t buffer_size = 1;
  auto ch = MakeChannel<int>(buffer_size);
  ChannelDestroyUnblockSenders(ch, true);
}

// This tests that destroying an unbuffered channel also unblocks
//  unblocks any receivers waiting for senders
TEST(Channel, UnbufferedChannelDestroyUnblocksReceiversTest) {
  auto ch = MakeChannel<int>(0);
  ChannelDestroyUnblockReceivers(ch);
}

TEST(Channel, UnbufferedChannelDestroyUnblocksSendersTest) {
  auto ch = MakeChannel<int>(0);
  ChannelDestroyUnblockSenders(ch, false);
}
