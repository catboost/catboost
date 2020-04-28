#pragma once

#include "node.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct IYsonConsumer;

void Serialize(const TString& value, IYsonConsumer* consumer);
void Serialize(const TStringBuf& value, IYsonConsumer* consumer);
void Serialize(const char* value, IYsonConsumer* consumer);
void Deserialize(TString& value, const TNode& node);

void Serialize(signed char value, IYsonConsumer* consumer);
void Serialize(short value, IYsonConsumer* consumer);
void Serialize(int value, IYsonConsumer* consumer);
void Serialize(long value, IYsonConsumer* consumer);
void Serialize(long long value, IYsonConsumer* consumer);
void Deserialize(i64& value, const TNode& node);

void Serialize(unsigned char value, IYsonConsumer* consumer);
void Serialize(unsigned short value, IYsonConsumer* consumer);
void Serialize(unsigned int value, IYsonConsumer* consumer);
void Serialize(unsigned long value, IYsonConsumer* consumer);
void Serialize(unsigned long long value, IYsonConsumer* consumer);
void Deserialize(ui64& value, const TNode& node);

void Serialize(double value, IYsonConsumer* consumer);
void Deserialize(double& value, const TNode& node);

void Serialize(bool value, IYsonConsumer* consumer);
void Deserialize(bool& value, const TNode& node);

void Serialize(const TNode& node, IYsonConsumer* consumer);
void Deserialize(TNode& value, const TNode& node);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
