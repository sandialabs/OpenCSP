// Copyright Sandia National Laboratories. All rights reserved.

#include "io/csv_reader.h"
#include "util/math.h"


/*
 * See https://stackoverflow.com/a/1120224
*/

class CsvRow
{
public:
    std::string_view operator[](std::size_t index) const
    {
        return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] - (m_data[index] + 1));
    }

    std::size_t size() const
    {
        return m_data.size() - 1;
    }

    void readNextRow(std::istream& srm)
    {
        std::getline(srm, m_line);

        m_data.clear();
        m_data.emplace_back(-1);
        std::string::size_type pos = 0;
        while ((pos = m_line.find(',', pos)) != std::string::npos)
        {
            m_data.emplace_back(pos);
            ++pos;
        }
        // check for a trailing comma
        pos = m_line.size();
        m_data.emplace_back(pos);
    }

private:
    std::string m_line;
    std::vector<int> m_data;
};


std::istream& operator>>(std::istream& srm, CsvRow& data)
{
    data.readNextRow(srm);
    return srm;
}


class CsvIterator
{
public:
    typedef std::input_iterator_tag iterator_category;
    typedef CsvRow value_type;
    typedef std::size_t difference_type;
    typedef CsvRow* pointer;
    typedef CsvRow& reference;

    CsvIterator(std::istream& srm) :_srm(srm.good() ? &srm : nullptr)
    {
        ++(*this);
    }

    CsvIterator() :_srm(nullptr) {}

    // Pre Increment
    CsvIterator& operator++()
    {
        if (_srm)
        {
            if (!((*_srm) >> _row))
            {
                _srm = nullptr;
            }
        }
        return *this;
    }

    // Post increment
    CsvIterator operator++(int)
    {
        CsvIterator tmp(*this);
        ++(*this);
        return tmp;
    }

    CsvRow const& operator*() const
    {
        return _row;
    }

    CsvRow const* operator->() const
    {
        return &_row;
    }

    bool operator==(CsvIterator const& rhs)
    {
        return (
            (this == &rhs) ||
            (
                (this->_srm == nullptr) &&
                (rhs._srm == nullptr)
            )
        );
    }

    bool operator!=(CsvIterator const& rhs)
    {
        return !((*this) == rhs);
    }

private:
    std::istream* _srm;
    CsvRow _row;
};


class CsvRange
{
    std::istream& stream;

public:
    CsvRange(std::istream& srm) : stream(srm) { }
    CsvIterator begin() const
    {
        return CsvIterator{ stream };
    }
    CsvIterator end() const
    {
        return CsvIterator{};
    }
};


inline float CellToFloat(std::string_view cell)
{
    char* end;
    float result = std::strtof(cell.data(), &end);
    // strtof stops at \r from Windows line endings, not a parse error
    const char* expected = cell.data() + cell.size();
    if (end != expected && !(end + 1 == expected && *end == '\r'))
    {
        result = 0.0f;
        std::cout << "Parse error\n";
    }
    return result;
}


// Loads heliostat positions and facet design from CSV; returns number of heliostats imported.
int ImportHeliostatData(const std::string& filename, float3* helioLocs, heliostat_design* design)
{
    //std::string fpath = "data\\" + filename;
    std::string fpath = filename;
    std::ifstream file(fpath);

    if (!file.is_open())
    {
        std::printf("failed to open %s\n", filename.c_str());
        return 0;
    }

    int nRow = 0, idx = 0;
    //int id = 0;

    for (auto& row : CsvRange(file))
    {
        // skip header row
        if (nRow == 0)
        {
            nRow++;
            continue;
        }

        float3& pos = helioLocs[idx];
        //id = (int)CellToFloat(row[0]);
        pos.x = CellToFloat(row[1]);
        pos.y = CellToFloat(row[2]);
        pos.z = CellToFloat(row[3]);

        // get facet data once; assume same for all heliostats
        if (nRow == 1)
        {
            design->nFacets = static_cast<int>(CellToFloat(row[4]));
            design->nRows = static_cast<int>(CellToFloat(row[5]));
            design->nCols = static_cast<int>(CellToFloat(row[6]));
            design->pivotHeight = CellToFloat(row[7]);
            design->pivotOffset = CellToFloat(row[8]);
            design->facetWidth = CellToFloat(row[9]);
            design->facetHeight = CellToFloat(row[10]);
        }

        nRow++;
        idx++;
    }

    std::printf("Import complete - %d heliostats imported.\n", idx);
    file.close();
    return idx;
}


// Loads per-heliostat azimuth/elevation aim angles from CSV and converts to unit vectors.
int ImportAimData(const std::string& filename, float3* aimVectors)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::printf("failed to open %s\n", filename.c_str());
        return 0;
    }

    int nRow = 0, idx = 0;
    //int id = 0;

    for (auto& row : CsvRange(file))
    {
        // skip header row
        if (nRow == 0)
        {
            nRow++;
            continue;
        }

        float azim = CellToFloat(row[1]);
        float elev = CellToFloat(row[2]);

        float3& vector = aimVectors[idx];
        vector = ConvertAzElToCartesian(azim, elev);
        std::cout << idx << " Aim vector: " << vector << std::endl;

        nRow++;
        idx++;
    }

    std::printf("Import complete - %d aim vector data rows imported.\n", idx);
    file.close();
    return idx;
}


// Loads facet centroid coordinates from CSV; returns number of facets imported.
int ImportFacetCoordinates(std::string filename, float3* facetOrigins)
{
    //std::string fpath = "data\\" + filename;
    std::string fpath = filename;
    std::ifstream file(fpath);

    if (!file.is_open())
    {
        std::printf("failed to open %s", filename.c_str());
        return 0;
    }
    int nRow = 0, id = 0, idx = 0;

    for (auto& row : CsvRange(file))
    {
        // skip header row
        if (nRow == 0)
        {
            nRow++;
            continue;
        }

        float3& pt = facetOrigins[idx];
        id = static_cast<int>(CellToFloat(row[0]));
        pt.x = CellToFloat(row[1]);
        pt.y = CellToFloat(row[2]);
        pt.z = CellToFloat(row[3]);
        //std::cout << fmt::format("{:d}: {:.4f}, {:.4f}, {:.4f}\n", id, pt.x, pt.y, pt.z);

        nRow++;
        idx++;
    }

    std::printf("Import complete - %d rows imported.\n", idx);
    file.close();
    return idx;
}


// ============================================================================
// Safe versions (validate inputs, resolve paths, return error codes)
// ============================================================================

ffiam::ImportResult ImportHeliostatDataSafe(const std::string& filename,
                                            float3* helioLocs,
                                            heliostat_design* design)
{
    using namespace ffiam;

    if (!helioLocs) {
        LogError(__func__, "helioLocs is null");
        return {ErrorCode::InvalidParameter, "helioLocs is null", 0};
    }
    if (!design) {
        LogError(__func__, "design is null");
        return {ErrorCode::InvalidParameter, "design is null", 0};
    }

    auto path = paths::ResolvePath(filename);
    if (!paths::FileExists(path)) {
        std::string msg = "File not found: " + path.string();
        LogError(__func__, msg);
        return {ErrorCode::FileNotFound, msg, 0};
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        std::string msg = "Failed to open: " + path.string();
        LogError(__func__, msg);
        return {ErrorCode::FileNotFound, msg, 0};
    }

    int nRow = 0, idx = 0;

    for (auto& row : CsvRange(file)) {
        if (nRow == 0) {
            nRow++;
            continue;
        }

        if (row.size() < 11) {
            LogWarning(__func__, "Row has insufficient columns, skipping");
            nRow++;
            continue;
        }

        float3& pos = helioLocs[idx];
        pos.x = CellToFloat(row[1]);
        pos.y = CellToFloat(row[2]);
        pos.z = CellToFloat(row[3]);

        if (nRow == 1) {
            design->nFacets = static_cast<int>(CellToFloat(row[4]));
            design->nRows = static_cast<int>(CellToFloat(row[5]));
            design->nCols = static_cast<int>(CellToFloat(row[6]));
            design->pivotHeight = CellToFloat(row[7]);
            design->pivotOffset = CellToFloat(row[8]);
            design->facetWidth = CellToFloat(row[9]);
            design->facetHeight = CellToFloat(row[10]);
        }

        nRow++;
        idx++;
    }

    file.close();
    std::printf("Import complete - %d heliostats imported.\n", idx);
    return {ErrorCode::Success, "", idx};
}


ffiam::ImportResult ImportAimDataSafe(const std::string& filename,
                                      float3* aimVectors)
{
    using namespace ffiam;

    if (!aimVectors) {
        LogError(__func__, "aimVectors is null");
        return {ErrorCode::InvalidParameter, "aimVectors is null", 0};
    }

    auto path = paths::ResolvePath(filename);
    if (!paths::FileExists(path)) {
        std::string msg = "File not found: " + path.string();
        LogError(__func__, msg);
        return {ErrorCode::FileNotFound, msg, 0};
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        std::string msg = "Failed to open: " + path.string();
        LogError(__func__, msg);
        return {ErrorCode::FileNotFound, msg, 0};
    }

    int nRow = 0, idx = 0;

    for (auto& row : CsvRange(file)) {
        if (nRow == 0) {
            nRow++;
            continue;
        }

        if (row.size() < 3) {
            LogWarning(__func__, "Row has insufficient columns, skipping");
            nRow++;
            continue;
        }

        float azim = CellToFloat(row[1]);
        float elev = CellToFloat(row[2]);

        float3& vector = aimVectors[idx];
        vector = ConvertAzElToCartesian(azim, elev);

        nRow++;
        idx++;
    }

    file.close();
    std::printf("Import complete - %d aim vector data rows imported.\n", idx);
    return {ErrorCode::Success, "", idx};
}


ffiam::ImportResult ImportFacetCoordinatesSafe(const std::string& filename,
                                               float3* facetOrigins)
{
    using namespace ffiam;

    if (!facetOrigins) {
        LogError(__func__, "facetOrigins is null");
        return {ErrorCode::InvalidParameter, "facetOrigins is null", 0};
    }

    auto path = paths::ResolvePath(filename);
    if (!paths::FileExists(path)) {
        std::string msg = "File not found: " + path.string();
        LogError(__func__, msg);
        return {ErrorCode::FileNotFound, msg, 0};
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        std::string msg = "Failed to open: " + path.string();
        LogError(__func__, msg);
        return {ErrorCode::FileNotFound, msg, 0};
    }

    int nRow = 0, idx = 0;

    for (auto& row : CsvRange(file)) {
        if (nRow == 0) {
            nRow++;
            continue;
        }

        if (row.size() < 4) {
            LogWarning(__func__, "Row has insufficient columns, skipping");
            nRow++;
            continue;
        }

        float3& pt = facetOrigins[idx];
        pt.x = CellToFloat(row[1]);
        pt.y = CellToFloat(row[2]);
        pt.z = CellToFloat(row[3]);

        nRow++;
        idx++;
    }

    file.close();
    std::printf("Import complete - %d rows imported.\n", idx);
    return {ErrorCode::Success, "", idx};
}
