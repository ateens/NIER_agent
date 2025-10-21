from fastmcp import FastMCP, settings as fastmcp_settings
from starlette.responses import JSONResponse

mcp = FastMCP("nier_analyzer")

def E2M3_fetch():
    return "1, 2, 3, 4, 5, 6, 7"


@mcp.tool(description="측정소 ID와 기간, 원소명을 입력하면 해당 데이터를 가져옵니다")
def fetch_timeseries(station_id: int, start_time, end_time, element):
    # 1. 기준측정소, 주변측정소 데이터 가져온다
    data = E2M3_fetch()
    # 2. (1) 을 DTW 비교 해서  Score 만든다
    
    # 3. VectorDB 에서 유사 데이터 가져온다
    # 4. result <- 종합판정 결과 가져온다 
    
    result = f"MCP 서버에서 가져온 데이터입니다: 해당 기간 데이터는 정상으로 보입니다 {data}"
    return result


@mcp.custom_route("/", methods=["GET"])
async def manifest(_request):
    """LangFlow에서 MCP 서버 정보를 확인할 수 있도록 루트 Manifest 를 제공한다."""
    return JSONResponse(
        {
            "name": "nier_analyzer",
            "version": "0.1.0",
            "protocol": {
                "version": "1.0",
                "capabilities": {
                    "tools": True,
                    "resources": False,
                    "instructions": False,
                },
            },
            "transport": {
                "type": "sse",
                "url": fastmcp_settings.sse_path,
            },
        }
    )


if __name__ == "__main__":
    
    mcp.run(transport="sse", host="0.0.0.0", port=9999)
