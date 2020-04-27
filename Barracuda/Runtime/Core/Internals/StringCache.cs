using System;
using System.Collections;
using System.Collections.Generic;

namespace Unity.Barracuda
{

internal struct StringStringPair : IEquatable<StringStringPair>
{
    public string a;
    public string b;

    public bool Equals(StringStringPair other)
    {
        return string.Equals(a, other.a) && string.Equals(b, other.b);
    }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        return obj is StringStringPair && Equals((StringStringPair) obj);
    }

    public override int GetHashCode()
    {
        var hashCode = a.GetHashCode();
        hashCode ^= b.GetHashCode();
        return hashCode;
    }
}

internal struct StringStringLongTriplet : IEquatable<StringStringLongTriplet>
{
    public string a;
    public string b;
    public long c;

    public override int GetHashCode()
    {
        var hashCode = a.GetHashCode();
        hashCode ^= b.GetHashCode();
        hashCode ^= c.GetHashCode();
        return hashCode;
    }

    public bool Equals(StringStringLongTriplet other)
    {
        return string.Equals(a, other.a) && string.Equals(b, other.b) && c == other.c;
    }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        return obj is StringStringLongTriplet && Equals((StringStringLongTriplet) obj);
    }
}

public class StringCache
{
    private Dictionary<StringStringPair, string> m_CacheStringString = new Dictionary<StringStringPair, string>();
    private Dictionary<StringStringLongTriplet, string> m_CacheStringStringLong = new Dictionary<StringStringLongTriplet, string>();

    public string Lookup(string a, string b)
    {
        var key = new StringStringPair {a = a ?? "", b = b ?? ""};

        if (!m_CacheStringString.ContainsKey(key))
            m_CacheStringString[key] = a + b;

        return m_CacheStringString[key];
    }

    public string Lookup(string a, string b, long c)
    {
        var key = new StringStringLongTriplet {a = a ?? "", b = b ?? "", c = c};

        if (!m_CacheStringStringLong.ContainsKey(key))
            m_CacheStringStringLong[key] = a + b + c;

        return m_CacheStringStringLong[key];
    }

    public void Clear()
    {
        m_CacheStringString.Clear();
        m_CacheStringStringLong.Clear();
    }
}

} // namespace Unity.Barracuda
